"""
APScheduler wrapper for continuous ingestion pipeline execution.

Runs the ingestion pipeline on a configurable interval (default: every
6 hours) so new SEC filings, news, and market data are always indexed.

Usage:
    from ingestion.scheduler import start_scheduler, stop_scheduler
    scheduler = start_scheduler()
    # ... run your application ...
    stop_scheduler(scheduler)

Or run directly:
    python -m ingestion.scheduler
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from config import DEFAULT_WATCHLIST, PIPELINE_SCHEDULE_HOURS
from ingestion.pipeline import run_pipeline

logger = logging.getLogger(__name__)


def _pipeline_job() -> None:
    """Scheduled job: run the full ingestion pipeline."""
    logger.info("Scheduled ingestion pipeline starting.")
    results = run_pipeline(watchlist=DEFAULT_WATCHLIST)
    total_filing_chunks = sum(r.get("filings_inserted", 0) for r in results)
    total_news_chunks = sum(r.get("news_inserted", 0) for r in results)
    errors = [e for r in results for e in r.get("errors", [])]
    logger.info(
        "Scheduled pipeline complete: %d filing chunks, %d news chunks. Errors: %d",
        total_filing_chunks,
        total_news_chunks,
        len(errors),
    )
    if errors:
        for err in errors:
            logger.warning("Pipeline error: %s", err)


def start_scheduler(interval_hours: int = PIPELINE_SCHEDULE_HOURS) -> BackgroundScheduler:
    """
    Start the background scheduler and return it.

    The caller is responsible for calling stop_scheduler() on shutdown.
    """
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _pipeline_job,
        trigger=IntervalTrigger(hours=interval_hours),
        id="ingestion_pipeline",
        name="Ingestion Pipeline",
        replace_existing=True,
        max_instances=1,  # prevent overlapping runs
    )
    scheduler.start()
    logger.info(
        "Ingestion pipeline scheduler started — runs every %d hours.", interval_hours
    )
    return scheduler


def stop_scheduler(scheduler: BackgroundScheduler) -> None:
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Ingestion pipeline scheduler stopped.")


if __name__ == "__main__":
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Running pipeline once immediately, then scheduling.")
    _pipeline_job()

    sched = start_scheduler()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        stop_scheduler(sched)
        logger.info("Scheduler shut down.")
