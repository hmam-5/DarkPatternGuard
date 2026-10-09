# Frontend

The browser-based DarkPatternGuard demonstration UI belongs here.

## Development

Install dependencies and start Vite from this directory:

```powershell
npm install
npm run dev
```

The API defaults to `http://localhost:8000`; set `VITE_API_URL` to override
it. Production validation is `npm run build`, and the mocked browser smoke
test is `npm run test:e2e` (install Playwright browsers with
`npx playwright install` when needed).
