"""
OCR Background Worker

Processes OCR jobs from the queue and delegates to specialized agents.
"""

import logging
import os
from typing import Any, Dict
from pathlib import Path

from ocr.workers.job_queue import JobQueue, Job
from ocr.communication.rabbitmq_transport import RabbitMQTransport

logger = logging.getLogger(__name__)


class OCRWorker:
    """
    Background worker for OCR jobs.

    Connects to job queue and processes OCR tasks by delegating
    to specialized agents via RabbitMQ.
    """

    def __init__(
        self,
        worker_id: str = "worker.ocr.1",
        queue_name: str = "jobs.ocr",
        rabbitmq_host: str = "rabbitmq"
    ):
        """
        Initialize OCR worker.

        Args:
            worker_id: Unique worker identifier
            queue_name: Job queue name
            rabbitmq_host: RabbitMQ server host
        """
        self.worker_id = worker_id
        self.queue_name = queue_name
        self.rabbitmq_host = rabbitmq_host

        # Initialize job queue
        self.job_queue = JobQueue(
            queue_name=queue_name,
            rabbitmq_host=rabbitmq_host
        )

        # Initialize transport for agent communication
        self.transport = RabbitMQTransport(
            host=rabbitmq_host,
            agent_id=worker_id
        )

        # Register job handlers
        self._register_handlers()

        logger.info(f"Initialized OCR worker: {worker_id}")

    def _register_handlers(self):
        """Register handlers for different job types."""
        self.job_queue.register_handler("ocr.preprocess", self._handle_preprocess_job)
        self.job_queue.register_handler("ocr.inference", self._handle_inference_job)
        self.job_queue.register_handler("ocr.validation", self._handle_validation_job)
        self.job_queue.register_handler("ocr.workflow", self._handle_workflow_job)
        self.job_queue.register_handler("ocr.batch", self._handle_batch_job)

    def _handle_preprocess_job(self, job: Job) -> Dict[str, Any]:
        """Handle preprocessing job."""
        logger.info(f"Processing preprocessing job {job.job_id}")

        payload = job.payload
        image_path = payload.get("image_path")
        options = payload.get("options", {})

        if not image_path:
            raise ValueError("image_path is required in payload")

        try:
            # Send command to preprocessing agent
            response = self.transport.send_command(
                target="agent.ocr.preprocessor",
                command="normalize_image",
                payload={
                    "image_path": image_path,
                    "options": options
                },
                timeout=60
            )

            return response.get("payload", {})

        except Exception as e:
            logger.error(f"Preprocessing job failed: {e}")
            raise

    def _handle_inference_job(self, job: Job) -> Dict[str, Any]:
        """Handle OCR inference job."""
        logger.info(f"Processing inference job {job.job_id}")

        payload = job.payload
        image_path = payload.get("image_path")
        model_config = payload.get("model_config", {})

        if not image_path:
            raise ValueError("image_path is required in payload")

        try:
            # Send command to inference agent
            response = self.transport.send_command(
                target="agent.ocr.inference",
                command="full_ocr",
                payload={
                    "image_path": image_path,
                    **model_config
                },
                timeout=120
            )

            return response.get("payload", {})

        except Exception as e:
            logger.error(f"Inference job failed: {e}")
            raise

    def _handle_validation_job(self, job: Job) -> Dict[str, Any]:
        """Handle validation job."""
        logger.info(f"Processing validation job {job.job_id}")

        payload = job.payload
        ocr_result = payload.get("ocr_result")
        validation_rules = payload.get("validation_rules", {})

        if not ocr_result:
            raise ValueError("ocr_result is required in payload")

        try:
            # Send command to validation agent
            response = self.transport.send_command(
                target="agent.ocr.validator",
                command="validate_ocr_result",
                payload={
                    "ocr_result": ocr_result,
                    "validation_rules": validation_rules,
                    "use_llm": validation_rules.get("use_llm", True)
                },
                timeout=90
            )

            return response.get("payload", {})

        except Exception as e:
            logger.error(f"Validation job failed: {e}")
            raise

    def _handle_workflow_job(self, job: Job) -> Dict[str, Any]:
        """Handle complete OCR workflow job."""
        logger.info(f"Processing workflow job {job.job_id}")

        payload = job.payload
        image_paths = payload.get("image_paths")
        workflow_config = payload.get("workflow_config", {})
        output_dir = payload.get("output_dir")

        if not image_paths:
            raise ValueError("image_paths is required in payload")

        try:
            # Send command to orchestrator agent
            response = self.transport.send_command(
                target="agent.orchestrator",
                command="execute_ocr_workflow",
                payload={
                    "image_paths": image_paths,
                    "workflow_config": workflow_config,
                    "output_dir": output_dir
                },
                timeout=600  # Longer timeout for full workflows
            )

            return response.get("payload", {})

        except Exception as e:
            logger.error(f"Workflow job failed: {e}")
            raise

    def _handle_batch_job(self, job: Job) -> Dict[str, Any]:
        """Handle batch OCR job."""
        logger.info(f"Processing batch job {job.job_id}")

        payload = job.payload
        image_dir = payload.get("image_dir")
        output_dir = payload.get("output_dir")
        batch_size = payload.get("batch_size", 10)

        if not image_dir:
            raise ValueError("image_dir is required in payload")

        try:
            # Discover images in directory
            from glob import glob

            image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
            image_paths = []

            for pattern in image_patterns:
                image_paths.extend(
                    glob(os.path.join(image_dir, "**", pattern), recursive=True)
                )

            if not image_paths:
                return {
                    "status": "success",
                    "message": "No images found in directory",
                    "processed": 0
                }

            logger.info(f"Found {len(image_paths)} images in {image_dir}")

            # Process in batches
            results = []
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]

                # Submit workflow for batch
                response = self.transport.send_command(
                    target="agent.orchestrator",
                    command="execute_ocr_workflow",
                    payload={
                        "image_paths": batch,
                        "workflow_config": payload.get("workflow_config", {}),
                        "output_dir": output_dir
                    },
                    timeout=600
                )

                results.append(response.get("payload", {}))

            return {
                "status": "success",
                "total_images": len(image_paths),
                "batches": len(results),
                "results": results
            }

        except Exception as e:
            logger.error(f"Batch job failed: {e}")
            raise

    def start(self, prefetch_count: int = 1):
        """
        Start the worker.

        Args:
            prefetch_count: Number of jobs to prefetch from queue
        """
        logger.info(f"Starting OCR worker {self.worker_id}...")

        try:
            # Connect transport for agent communication
            self.transport.connect()

            # Start processing jobs
            self.job_queue.start_worker(prefetch_count=prefetch_count)

        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
        except Exception as e:
            logger.error(f"Worker error: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self):
        """Stop the worker and cleanup resources."""
        logger.info(f"Stopping OCR worker {self.worker_id}...")
        self.job_queue.close()
        self.transport.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Start worker
    worker = OCRWorker()
    worker.start()
