-- Safe to apply to the existing JobGPT database before enabling backfill.
alter table public.jobs
    add column if not exists description_excerpt text,
    add column if not exists details_checked_at timestamptz;
