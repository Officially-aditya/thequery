CREATE TABLE IF NOT EXISTS admin_login_rate_limits (
  key_hash TEXT PRIMARY KEY,
  attempts INTEGER NOT NULL CHECK (attempts > 0),
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS admin_login_rate_limits_expires_at_idx
  ON admin_login_rate_limits (expires_at);
