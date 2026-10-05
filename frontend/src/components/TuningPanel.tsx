/**
 * Developer tuning panel for viewing game configuration.
 */

import type { GameConfig } from '../game/types'

interface TuningPanelProps {
  config: GameConfig
  onClose: () => void
}

export function TuningPanel({ config, onClose }: TuningPanelProps) {
  const configItems = [
    { label: 'Pitch Width', value: config.pitch_width, unit: 'units' },
    { label: 'Pitch Height', value: config.pitch_height, unit: 'units' },
    { label: 'Goal Width', value: config.goal_width, unit: 'units' },
    { label: 'Goal Height', value: config.goal_height, unit: 'units' },
    { label: 'Player Radius', value: config.player_radius, unit: 'units' },
    { label: 'Ball Radius', value: config.ball_radius, unit: 'units' },
  ]

  return (
    <div style={{
      width: '280px',
      background: 'linear-gradient(180deg, #2d2d44 0%, #1a1a2e 100%)',
      borderLeft: '1px solid #3d3d5c',
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px',
      overflowY: 'auto',
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <h2 style={{ fontSize: '18px', fontWeight: 600 }}>
          ⚙️ Configuration
        </h2>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            color: '#aaa',
            fontSize: '24px',
            cursor: 'pointer',
            padding: '4px',
          }}
        >
          ×
        </button>
      </div>

      <p style={{ fontSize: '12px', color: '#888' }}>
        Physics constants from the backend simulation.
        These are read-only and controlled by the server.
      </p>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
      }}>
        {configItems.map(({ label, value, unit }) => (
          <div key={label} style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '8px 12px',
            background: '#1a1a2e',
            borderRadius: '8px',
          }}>
            <span style={{ fontSize: '13px', color: '#ccc' }}>{label}</span>
            <span style={{ 
              fontSize: '14px', 
              fontWeight: 600,
              fontFamily: 'monospace',
            }}>
              {typeof value === 'number' ? value.toFixed(2) : value} 
              <span style={{ color: '#666', fontWeight: 400, marginLeft: '4px' }}>
                {unit}
              </span>
            </span>
          </div>
        ))}
      </div>

      {/* Team colors */}
      <div style={{ marginTop: '8px' }}>
        <h3 style={{ fontSize: '14px', marginBottom: '12px', color: '#aaa' }}>
          Team Colors
        </h3>
        <div style={{ display: 'flex', gap: '12px' }}>
          <div style={{
            flex: 1,
            padding: '12px',
            background: `rgb(${config.team_a_color.join(',')})`,
            borderRadius: '8px',
            textAlign: 'center',
            color: '#333',
            fontWeight: 600,
          }}>
            Team A
          </div>
          <div style={{
            flex: 1,
            padding: '12px',
            background: `rgb(${config.team_b_color.join(',')})`,
            borderRadius: '8px',
            textAlign: 'center',
            color: '#333',
            fontWeight: 600,
          }}>
            Team B
          </div>
        </div>
      </div>

      {/* Help */}
      <div style={{
        marginTop: 'auto',
        padding: '12px',
        background: '#1a1a2e',
        borderRadius: '8px',
        fontSize: '12px',
        color: '#888',
      }}>
        <strong style={{ color: '#aaa' }}>How to play:</strong>
        <ol style={{ paddingLeft: '16px', marginTop: '8px', lineHeight: 1.6 }}>
          <li>Click on one of your team's players</li>
          <li>Drag back (like a slingshot)</li>
          <li>Release to flick the player!</li>
        </ol>
      </div>
    </div>
  )
}
