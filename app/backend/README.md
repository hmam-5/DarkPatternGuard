# Backend

The inference API and server-side application code belong here.

Run the CPU serving API with:

```powershell
make app
```

Interactive OpenAPI documentation is available at `/docs`; model artifacts
are loaded once during startup and the registry falls back safely when trained
serving artifacts are not present.

The database uses SQLite by default. Set `DATABASE_URL` to a PostgreSQL URL
for deployment, then run `alembic upgrade head`. Prediction text can be
excluded from storage with `privacy_flag=true`.
