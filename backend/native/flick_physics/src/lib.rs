//! High-performance flick-soccer physics core (PyO3).
//! Mirrors backend/sim Python physics/collision semantics for RL throughput.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

const EPS: f64 = 1e-10;

#[derive(Clone, Copy)]
struct Body {
    x: f64,
    y: f64,
    vx: f64,
    vy: f64,
    r: f64,
    m: f64,
    sleeping: bool,
    sleep_timer: f64,
}

#[derive(Clone, Copy)]
struct Wall {
    x1: f64,
    y1: f64,
    dx: f64,
    dy: f64,
    len_sq: f64,
}

#[derive(Clone, Copy)]
struct Post {
    x: f64,
    y: f64,
    r: f64,
}

#[derive(Clone, Copy)]
struct Cfg {
    timestep: f64,
    max_substeps: i32,
    player_drag: f64,
    ball_drag: f64,
    restitution: f64,
    player_player_restitution: f64,
    wall_restitution: f64,
    post_restitution: f64,
    collision_friction: f64,
    max_launch_speed: f64,
    min_launch_speed: f64,
    sleep_threshold: f64,
    sleep_time_required: f64,
    ccd_threshold: f64,
    min_separation: f64,
    ball_radius: f64,
    max_sim_time: f64,
    goal_y_min: f64,
    goal_y_max: f64,
    pitch_width: f64,
}

fn unpack_body(slice: &[f64]) -> Body {
    Body {
        x: slice[0],
        y: slice[1],
        vx: slice[2],
        vy: slice[3],
        r: slice[4],
        m: slice[5],
        sleeping: slice[6] > 0.5,
        sleep_timer: slice[7],
    }
}

fn pack_body(b: &Body, out: &mut [f64]) {
    out[0] = b.x;
    out[1] = b.y;
    out[2] = b.vx;
    out[3] = b.vy;
    out[4] = b.r;
    out[5] = b.m;
    out[6] = if b.sleeping { 1.0 } else { 0.0 };
    out[7] = b.sleep_timer;
}

fn speed(vx: f64, vy: f64) -> f64 {
    (vx * vx + vy * vy).sqrt()
}

fn apply_drag(vx: f64, vy: f64, drag: f64, dt: f64) -> (f64, f64) {
    let f = (-drag * dt).exp();
    (vx * f, vy * f)
}

fn update_sleep(b: &mut Body, dt: f64, threshold: f64, time_required: f64) {
    let s = speed(b.vx, b.vy);
    if s < threshold {
        b.sleep_timer += dt;
        if b.sleep_timer >= time_required {
            b.sleeping = true;
            b.vx = 0.0;
            b.vy = 0.0;
        }
    } else {
        b.sleep_timer = 0.0;
        b.sleeping = false;
    }
}

fn step_body(b: &mut Body, drag: f64, dt: f64, cfg: &Cfg) {
    if b.sleeping {
        return;
    }
    let (vx, vy) = apply_drag(b.vx, b.vy, drag, dt);
    b.vx = vx;
    b.vy = vy;
    b.x += b.vx * dt;
    b.y += b.vy * dt;
    update_sleep(b, dt, cfg.sleep_threshold, cfg.sleep_time_required);
}

fn launch(b: &mut Body, dx: f64, dy: f64, power: f64, cfg: &Cfg) {
    let len = (dx * dx + dy * dy).sqrt();
    let (ndx, ndy) = if len < EPS {
        (1.0, 0.0)
    } else {
        (dx / len, dy / len)
    };
    let spd = cfg.min_launch_speed + (cfg.max_launch_speed - cfg.min_launch_speed) * power;
    b.vx = ndx * spd;
    b.vy = ndy * spd;
    b.sleeping = false;
    b.sleep_timer = 0.0;
}

fn substeps(speed: f64, radius: f64, dt: f64, cfg: &Cfg) -> i32 {
    if speed < cfg.ccd_threshold {
        return 1;
    }
    let max_step = radius * 0.5;
    let total = speed * dt;
    let n = (total / max_step).ceil() as i32;
    n.min(cfg.max_substeps).max(1)
}

fn impulse_response(
    _x1: f64,
    _y1: f64,
    vx1: f64,
    vy1: f64,
    m1: f64,
    _x2: f64,
    _y2: f64,
    vx2: f64,
    vy2: f64,
    m2: f64,
    nx: f64,
    ny: f64,
    restitution: f64,
    friction: f64,
) -> (f64, f64, f64, f64) {
    let rvx = vx2 - vx1;
    let rvy = vy2 - vy1;
    let vel_n = rvx * nx + rvy * ny;
    if vel_n > 0.0 {
        return (vx1, vy1, vx2, vy2);
    }
    let inv1 = 1.0 / m1;
    let inv2 = 1.0 / m2;
    let inv_sum = inv1 + inv2;
    let mut jn = -(1.0 + restitution) * vel_n / inv_sum;
    let mut ix = nx * jn;
    let mut iy = ny * jn;
    if friction > 0.0 {
        let tx = rvx - nx * vel_n;
        let ty = rvy - ny * vel_n;
        let tlen = (tx * tx + ty * ty).sqrt();
        if tlen > 1e-8 {
            let tnx = tx / tlen;
            let tny = ty / tlen;
            let mut jt = -(rvx * tnx + rvy * tny) / inv_sum;
            let max_jt = jn.abs() * friction;
            if jt > max_jt {
                jt = max_jt;
            } else if jt < -max_jt {
                jt = -max_jt;
            }
            ix += tnx * jt;
            iy += tny * jt;
        }
    }
    (
        vx1 - ix * inv1,
        vy1 - iy * inv1,
        vx2 + ix * inv2,
        vy2 + iy * inv2,
    )
}

fn separate_circles(
    x1: f64,
    y1: f64,
    r1: f64,
    m1: f64,
    x2: f64,
    y2: f64,
    r2: f64,
    m2: f64,
    min_sep: f64,
) -> (f64, f64, f64, f64) {
    let mut dx = x2 - x1;
    let mut dy = y2 - y1;
    let mut dist = (dx * dx + dy * dy).sqrt();
    if dist < EPS {
        dx = 1.0;
        dy = 0.0;
        dist = 0.0;
    }
    let overlap = r1 + r2 - dist + min_sep;
    if overlap <= 0.0 {
        return (x1, y1, x2, y2);
    }
    let inv = 1.0 / dist.max(EPS);
    let nx = dx * inv;
    let ny = dy * inv;
    let total = m1 + m2;
    let move1 = overlap * (m2 / total);
    let move2 = overlap * (m1 / total);
    (
        x1 - nx * move1,
        y1 - ny * move1,
        x2 + nx * move2,
        y2 + ny * move2,
    )
}

fn closest_on_seg(px: f64, py: f64, w: &Wall) -> (f64, f64) {
    if w.len_sq < EPS {
        return (w.x1, w.y1);
    }
    let mut t = ((px - w.x1) * w.dx + (py - w.y1) * w.dy) / w.len_sq;
    if t < 0.0 {
        t = 0.0;
    } else if t > 1.0 {
        t = 1.0;
    }
    (w.x1 + w.dx * t, w.y1 + w.dy * t)
}

fn seg_normal(w: &Wall) -> (f64, f64) {
    let len = w.len_sq.sqrt().max(1.0);
    (-w.dy / len, w.dx / len)
}

fn wall_impulse(vx: f64, vy: f64, nx: f64, ny: f64, restitution: f64) -> (f64, f64) {
    let vn = vx * nx + vy * ny;
    if vn >= 0.0 {
        return (vx, vy);
    }
    (
        vx - nx * (1.0 + restitution) * vn,
        vy - ny * (1.0 + restitution) * vn,
    )
}

fn hit_wall(b: &mut Body, walls: &[Wall], cfg: &Cfg) -> bool {
    if b.sleeping {
        return false;
    }
    let mut changed = false;
    for w in walls {
        let (cx, cy) = closest_on_seg(b.x, b.y, w);
        let ddx = b.x - cx;
        let ddy = b.y - cy;
        let dist = (ddx * ddx + ddy * ddy).sqrt();
        if dist >= b.r {
            continue;
        }
        let (nx, ny) = if dist < EPS {
            seg_normal(w)
        } else {
            (ddx / dist, ddy / dist)
        };
        let vel_toward = b.vx * nx + b.vy * ny;
        if vel_toward < 0.0 {
            let (vx, vy) = wall_impulse(b.vx, b.vy, nx, ny, cfg.wall_restitution);
            b.vx = vx;
            b.vy = vy;
            changed = true;
        }
        let required = b.r + cfg.min_separation;
        let dist_n = (b.x - cx) * nx + (b.y - cy) * ny;
        if dist_n < required {
            let push = required - dist_n;
            b.x += nx * push;
            b.y += ny * push;
            changed = true;
        }
        if changed {
            b.sleeping = false;
            b.sleep_timer = 0.0;
        }
    }
    changed
}

fn circle_hit(ax: f64, ay: f64, ar: f64, bx: f64, by: f64, br: f64) -> Option<(f64, f64, f64)> {
    let dx = bx - ax;
    let dy = by - ay;
    let dist_sq = dx * dx + dy * dy;
    let rsum = ar + br;
    if dist_sq >= rsum * rsum {
        return None;
    }
    let dist = dist_sq.sqrt();
    let (nx, ny) = if dist < EPS {
        (1.0, 0.0)
    } else {
        (dx / dist, dy / dist)
    };
    let pen = rsum - dist;
    Some((nx, ny, pen))
}

fn resolve_ball_players(ball: &mut Body, players: &mut [Body], cfg: &Cfg, last_touch: &mut i32) -> bool {
    let mut changed = false;
    for (i, p) in players.iter_mut().enumerate() {
        if ball.sleeping && p.sleeping {
            continue;
        }
        if let Some((nx, ny, _pen)) = circle_hit(ball.x, ball.y, ball.r, p.x, p.y, p.r) {
            let (bvx, bvy, pvx, pvy) = impulse_response(
                ball.x, ball.y, ball.vx, ball.vy, ball.m,
                p.x, p.y, p.vx, p.vy, p.m,
                nx, ny, cfg.restitution, cfg.collision_friction,
            );
            ball.vx = bvx;
            ball.vy = bvy;
            p.vx = pvx;
            p.vy = pvy;
            ball.sleeping = false;
            ball.sleep_timer = 0.0;
            p.sleeping = false;
            p.sleep_timer = 0.0;
            let (bx, by, px, py) = separate_circles(
                ball.x, ball.y, ball.r, ball.m,
                p.x, p.y, p.r, p.m,
                cfg.min_separation,
            );
            ball.x = bx;
            ball.y = by;
            p.x = px;
            p.y = py;
            *last_touch = i as i32;
            changed = true;
        }
    }
    changed
}

fn resolve_player_player(players: &mut [Body], cfg: &Cfg) -> bool {
    let mut changed = false;
    let n = players.len();
    for i in 0..n {
        for j in (i + 1)..n {
            // split borrow
            let (left, right) = players.split_at_mut(j);
            let p1 = &mut left[i];
            let p2 = &mut right[0];
            if p1.sleeping && p2.sleeping {
                continue;
            }
            if let Some((nx, ny, _)) = circle_hit(p1.x, p1.y, p1.r, p2.x, p2.y, p2.r) {
                let (v1x, v1y, v2x, v2y) = impulse_response(
                    p1.x, p1.y, p1.vx, p1.vy, p1.m,
                    p2.x, p2.y, p2.vx, p2.vy, p2.m,
                    nx, ny, cfg.player_player_restitution, cfg.collision_friction,
                );
                p1.vx = v1x;
                p1.vy = v1y;
                p2.vx = v2x;
                p2.vy = v2y;
                p1.sleeping = false;
                p1.sleep_timer = 0.0;
                p2.sleeping = false;
                p2.sleep_timer = 0.0;
                let (x1, y1, x2, y2) = separate_circles(
                    p1.x, p1.y, p1.r, p1.m,
                    p2.x, p2.y, p2.r, p2.m,
                    cfg.min_separation,
                );
                p1.x = x1;
                p1.y = y1;
                p2.x = x2;
                p2.y = y2;
                changed = true;
            }
        }
    }
    changed
}

fn resolve_posts(ball: &mut Body, players: &mut [Body], posts: &[Post], cfg: &Cfg) -> bool {
    let mut changed = false;
    for post in posts {
        if !ball.sleeping {
            if let Some((nx, ny, _)) = circle_hit(ball.x, ball.y, ball.r, post.x, post.y, post.r) {
                // normal from post to ball is -nx,-ny of (ball->post)? circle_hit is from a to b
                // Python: normal from ball to post, then vel_toward = ball.velocity.dot(-normal)
                let vel_toward = ball.vx * (-nx) + ball.vy * (-ny);
                if vel_toward > 0.0 {
                    let (vx, vy) = wall_impulse(ball.vx, ball.vy, -nx, -ny, cfg.post_restitution);
                    ball.vx = vx;
                    ball.vy = vy;
                    ball.sleeping = false;
                    ball.sleep_timer = 0.0;
                    changed = true;
                }
                let (bx, by, _, _) = separate_circles(
                    ball.x, ball.y, ball.r, ball.m,
                    post.x, post.y, post.r, 1e10,
                    cfg.min_separation,
                );
                ball.x = bx;
                ball.y = by;
                changed = true;
            }
        }
        for p in players.iter_mut() {
            if p.sleeping {
                continue;
            }
            if let Some((nx, ny, _)) = circle_hit(p.x, p.y, p.r, post.x, post.y, post.r) {
                let vel_toward = p.vx * (-nx) + p.vy * (-ny);
                if vel_toward > 0.0 {
                    let (vx, vy) = wall_impulse(p.vx, p.vy, -nx, -ny, cfg.post_restitution);
                    p.vx = vx;
                    p.vy = vy;
                    p.sleeping = false;
                    p.sleep_timer = 0.0;
                    changed = true;
                }
                let (px, py, _, _) = separate_circles(
                    p.x, p.y, p.r, p.m,
                    post.x, post.y, post.r, 1e10,
                    cfg.min_separation,
                );
                p.x = px;
                p.y = py;
                changed = true;
            }
        }
    }
    changed
}

fn resolve_all(
    ball: &mut Body,
    players: &mut [Body],
    walls: &[Wall],
    posts: &[Post],
    cfg: &Cfg,
    last_touch: &mut i32,
) {
    for _ in 0..4 {
        let mut changed = false;
        changed |= resolve_ball_players(ball, players, cfg, last_touch);
        changed |= resolve_player_player(players, cfg);
        changed |= hit_wall(ball, walls, cfg);
        for p in players.iter_mut() {
            changed |= hit_wall(p, walls, cfg);
        }
        changed |= resolve_posts(ball, players, posts, cfg);
        if !changed {
            break;
        }
    }
}

fn check_goal(ball: &Body, cfg: &Cfg) -> i32 {
    // 1 = Team B scored (left goal), 2 = Team A scored (right goal), 0 = none
    if ball.x < 0.0 && ball.y > cfg.goal_y_min && ball.y < cfg.goal_y_max {
        return 1;
    }
    if ball.x > cfg.pitch_width && ball.y > cfg.goal_y_min && ball.y < cfg.goal_y_max {
        return 2;
    }
    0
}

fn all_at_rest(ball: &Body, players: &[Body], threshold: f64) -> bool {
    if speed(ball.vx, ball.vy) >= threshold && !ball.sleeping {
        return false;
    }
    // Match Python: all_at_rest uses velocity length < threshold
    if speed(ball.vx, ball.vy) >= threshold {
        return false;
    }
    for p in players {
        if speed(p.vx, p.vy) >= threshold {
            return false;
        }
    }
    true
}

fn parse_walls(data: &[f64]) -> PyResult<Vec<Wall>> {
    if data.len() % 4 != 0 {
        return Err(PyValueError::new_err("walls must be multiple of 4"));
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let dx = chunk[2] - chunk[0];
        let dy = chunk[3] - chunk[1];
        out.push(Wall {
            x1: chunk[0],
            y1: chunk[1],
            dx,
            dy,
            len_sq: dx * dx + dy * dy,
        });
    }
    Ok(out)
}

fn parse_posts(data: &[f64]) -> PyResult<Vec<Post>> {
    if data.len() % 3 != 0 {
        return Err(PyValueError::new_err("posts must be multiple of 3"));
    }
    Ok(data
        .chunks_exact(3)
        .map(|c| Post {
            x: c[0],
            y: c[1],
            r: c[2],
        })
        .collect())
}

fn parse_cfg(c: &[f64]) -> PyResult<Cfg> {
    if c.len() < 20 {
        return Err(PyValueError::new_err("cfg needs >= 20 floats"));
    }
    Ok(Cfg {
        timestep: c[0],
        max_substeps: c[1] as i32,
        player_drag: c[2],
        ball_drag: c[3],
        restitution: c[4],
        player_player_restitution: c[5],
        wall_restitution: c[6],
        post_restitution: c[7],
        collision_friction: c[8],
        max_launch_speed: c[9],
        min_launch_speed: c[10],
        sleep_threshold: c[11],
        sleep_time_required: c[12],
        ccd_threshold: c[13],
        min_separation: c[14],
        ball_radius: c[15],
        max_sim_time: c[16],
        goal_y_min: c[17],
        goal_y_max: c[18],
        pitch_width: c[19],
    })
}

/// Simulate one flick until rest or goal.
/// Returns (ball8, players_flat, sim_time, goal_code, last_touch_idx).
#[pyfunction]
fn simulate_flick(
    ball_in: Vec<f64>,
    players_in: Vec<f64>,
    walls_in: Vec<f64>,
    posts_in: Vec<f64>,
    cfg_in: Vec<f64>,
    launch_idx: usize,
    launch_dx: f64,
    launch_dy: f64,
    power: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, f64, i32, i32)> {
    if ball_in.len() != 8 {
        return Err(PyValueError::new_err("ball must be length 8"));
    }
    if players_in.len() % 8 != 0 {
        return Err(PyValueError::new_err("players must be multiple of 8"));
    }
    let cfg = parse_cfg(&cfg_in)?;
    let walls = parse_walls(&walls_in)?;
    let posts = parse_posts(&posts_in)?;
    let mut ball = unpack_body(&ball_in);
    let n = players_in.len() / 8;
    let mut players: Vec<Body> = (0..n)
        .map(|i| unpack_body(&players_in[i * 8..(i + 1) * 8]))
        .collect();
    if launch_idx >= n {
        return Err(PyValueError::new_err("launch_idx out of range"));
    }
    launch(&mut players[launch_idx], launch_dx, launch_dy, power, &cfg);
    let mut last_touch = launch_idx as i32;
    let mut sim_time = 0.0;
    let dt = cfg.timestep;

    while sim_time < cfg.max_sim_time {
        let mut max_speed = speed(ball.vx, ball.vy);
        for p in &players {
            let s = speed(p.vx, p.vy);
            if s > max_speed {
                max_speed = s;
            }
        }
        let nsub = substeps(max_speed, cfg.ball_radius, dt, &cfg);
        let sub_dt = dt / nsub as f64;
        for _ in 0..nsub {
            step_body(&mut ball, cfg.ball_drag, sub_dt, &cfg);
            for p in players.iter_mut() {
                step_body(p, cfg.player_drag, sub_dt, &cfg);
            }
            resolve_all(&mut ball, &mut players, &walls, &posts, &cfg, &mut last_touch);
        }
        sim_time += dt;
        let goal = check_goal(&ball, &cfg);
        if goal != 0 {
            let mut bout = vec![0.0; 8];
            pack_body(&ball, &mut bout);
            let mut pout = vec![0.0; n * 8];
            for (i, p) in players.iter().enumerate() {
                pack_body(p, &mut pout[i * 8..(i + 1) * 8]);
            }
            return Ok((bout, pout, sim_time, goal, last_touch));
        }
        if all_at_rest(&ball, &players, cfg.sleep_threshold) {
            break;
        }
    }

    let mut bout = vec![0.0; 8];
    pack_body(&ball, &mut bout);
    let mut pout = vec![0.0; n * 8];
    for (i, p) in players.iter().enumerate() {
        pack_body(p, &mut pout[i * 8..(i + 1) * 8]);
    }
    Ok((bout, pout, sim_time, 0, last_touch))
}

#[pymodule]
fn flick_physics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_flick, m)?)?;
    Ok(())
}
