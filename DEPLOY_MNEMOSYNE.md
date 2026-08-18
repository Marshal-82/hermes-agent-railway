# Deploy: Mnemosyne memory provider (the mem0-off switch)

This patch bakes the Odaro Memory Engine + Hermes memory-provider plugin
into the hermes-agent-railway image. It is INERT until activated.

## Activate (when ready)
1. Push this branch/commit to Marshal-82/hermes-agent-railway (Railway
   auto-deploys main).
2. On the hermes-agent-railway Railway service set:
     MNEMOSYNE_ENABLED=true
     MNEMOSYNE_DATABASE_URL=<supabase-pooler-url>   # Postgres w/ schema
3. Run the schema once (operator-only):
     DATABASE_URL=<same-url> python scripts/init_pg.py
   (from odaro-memory-engine; creates tables + RLS default-deny)
4. `hermes memory status` on the container should show mnemosyne active.
   prefetch() now feeds the model deterministic situation briefs — 0 LLM
   tokens on write/recall. mem0 can be retired.

Rollback: set MNEMOSYNE_ENABLED=false and redeploy (provider untouched).
