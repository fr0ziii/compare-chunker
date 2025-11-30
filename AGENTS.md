# Repository Guidelines

## Project Structure & Module Organization
The app is a Vite + React TypeScript workspace. User-facing code lives in `src/`, with `main.tsx` bootstrapping React and `App.tsx` holding the chunk comparison workflow (LangChain splitters, drag/drop ingest, highlighting). Shared assets go in `src/assets/`, while global styles live in `src/App.css` and `src/index.css`. Static files required during build (favicons, manifest) belong in `public/`. Production bundles land in `dist/` after `npm run build`; never edit files there manually. The `logs/` folder records local session payloads; treat it as disposable debugging data.

## Build, Test, and Development Commands
- `npm install`: sync dependencies defined in `package-lock.json`; rerun whenever `package.json` changes.
- `npm run dev`: start Vite with fast-refresh. Default port is 5173; use this to manually vet chunk ranges.
- `npm run build`: type-check via `tsc -b` then emit optimized Vite output into `dist/`.
- `npm run preview`: serve the most recent build to confirm production behavior.
- `npm run lint`: run the flat ESLint config (`eslint.config.js`) with React Hooks, Refresh, and TS rules.

## Coding Style & Naming Conventions
Code is TypeScript-first; enable strict types before merging. Use 2-space indentation, double quotes, and trailing commas where ESLint permits. Components are PascalCase (`ChunkCardList`), hooks/utilities camelCase, and boolean flags read positively (`isLoading`). Keep splitter helpers pure and colocate UI-specific logic inside components. Run Prettier via your editor, but rely on ESLint to catch deviations.

## Testing Guidelines
Automated tests are not yet wired in this template. When adding them, prefer Vitest + @testing-library/react, place files under `src/__tests__` or alongside modules with `.test.tsx`, and cover chunk boundary calculations plus drag/drop handlers. Snapshot tests are discouraged; assert on text ranges and derived metadata like `cleanBoundary`. Until a test script lands, document any manual verification steps inside PRs.

## Commit & Pull Request Guidelines
No Git metadata ships with this workspace, so adopt an imperative, Conventional-Commit-style summary (`feat: support CSV uploads`). Keep scope under 72 characters and include rationale in the body when touching chunk math or async flows. PRs should describe what changed, why, how to reproduce the bug or verify the feature, and include screenshots/gifs for UI adjustments plus references to issue IDs. Request review after `npm run lint` and `npm run build` succeed locally.

## Instructions
- Store every conversation as a JSON file inside `logs/`.
- When the user issues the command `save-conversation`, treat it as a request to persist the current conversation JSON and then end the session.
- Keep all JSON files consistent with the existing structure (session metadata + ordered message list with role/content).
- Optimize JSON for token efficiency (concise fields, avoid redundancy) to keep context windows lean.
