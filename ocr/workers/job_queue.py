"""
Job Queue System for Multi-Agent Collaboration

Provides a robust job queuing system on top of RabbitMQ for:
- Task distribution to background workers
- Job prioritization
- Retry logic
- Dead letter handling
"""

import logging
import json
import uuid
import time
from typing import Any, Optional, Callable, Dict
from enum import Enum
from dataclasses import dataclass, asdict
import pika

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


@dataclass
class Job:
    """Represents a job in the queue."""
    job_id: str
    job_type: str
    payload: Dict[str, Any]
    priority: int = JobPriority.NORMAL
    max_retries: int = 3
    retry_count: int = 0
    timeout: int = 300
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: JobStatus = JobStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        return cls(**data)


class JobQueue:
    """
    Job Queue system built on RabbitMQ.

    Provides high-level job queuing with priority, retries, and monitoring.
    """

    def __init__(
        self,
        queue_name: str,
        rabbitmq_host: str = "rabbitmq",
        exchange: str = "jobs.topic",
        dlx_exchange: str = "jobs.dlx"
    ):
        """
        Initialize job queue.

        Args:
            queue_name: Name of the job queue
            rabbitmq_host: RabbitMQ server host
            exchange: Main exchange name
            dlx_exchange: Dead letter exchange name
        """
        self.queue_name = queue_name
        self.host = rabbitmq_host
        self.exchange = exchange
        self.dlx_exchange = dlx_exchange

        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None

        self._job_handlers: Dict[str, Callable] = {}
        self._active_jobs: Dict[str, Job] = {}

    def connect(self):
        """Connect to RabbitMQ and setup queues."""
        try:
            logger.info(f"Connecting to RabbitMQ at {self.host}...")
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(host=self.host)
            )
            self.channel = self.connection.channel()

            # Declare main exchange
            self.channel.exchange_declare(
                exchange=self.exchange,
                exchange_type='topic',
                durable=True
            )

            # Declare dead letter exchange
            self.channel.exchange_declare(
                exchange=self.dlx_exchange,
                exchange_type='topic',
                durable=True
            )

            # Declare main queue with DLX
            self.channel.queue_declare(
                queue=self.queue_name,
                durable=True,
                arguments={
                    'x-dead-letter-exchange': self.dlx_exchange,
                    'x-dead-letter-routing-key': f'{self.queue_name}.dlq'
                }
            )

            # Declare dead letter queue
            dlq_name = f"{self.queue_name}.dlq"
            self.channel.queue_declare(queue=dlq_name, durable=True)
            self.channel.queue_bind(
                exchange=self.dlx_exchange,
                queue=dlq_name,
                routing_key=f'{self.queue_name}.dlq'
            )

            # Bind main queue to exchange
            self.channel.queue_bind(
                exchange=self.exchange,
                queue=self.queue_name,
                routing_key=f'{self.queue_name}.#'
            )

            logger.info(f"Connected to job queue: {self.queue_name}")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def close(self):
        """Close the connection."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
            logger.info("Connection closed")

    def submit_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: int = JobPriority.NORMAL,
        max_retries: int = 3,
        timeout: int = 300
    ) -> str:
        """
        Submit a job to the queue.

        Args:
            job_type: Type of job to execute
            payload: Job payload/parameters
            priority: Job priority (1-10)
            max_retries: Maximum retry attempts
            timeout: Timeout in seconds

        Returns:
            Job ID
        """
        if not self.connection:
            raise RuntimeError("Not connected to RabbitMQ")

        job = Job(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
            timeout=timeout
        )

        routing_key = f"{self.queue_name}.{job_type}"

        self.channel.basic_publish(
            exchange=self.exchange,
            routing_key=routing_key,
            body=json.dumps(job.to_dict()),
            properties=pika.BasicProperties(
                delivery_mode=2,  # Persistent
                priority=priority,
                content_type='application/json'
            )
        )

        logger.info(f"Submitted job {job.job_id} ({job_type}) with priority {priority}")
        return job.job_id

    def register_handler(self, job_type: str, handler: Callable[[Job], Dict[str, Any]]):
        """
        Register a handler for a job type.

        Args:
            job_type: Type of job to handle
            handler: Callable that processes the job and returns result
        """
        self._job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    def _process_job(self, job: Job) -> Dict[str, Any]:
        """
        Process a job using registered handler.

        Args:
            job: Job to process

        Returns:
            Job result
        """
        handler = self._job_handlers.get(job.job_type)

        if not handler:
            raise ValueError(f"No handler registered for job type: {job.job_type}")

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._active_jobs[job.job_id] = job

        try:
            result = handler(job)
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.result = result
            return result

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            logger.error(f"Job {job.job_id} failed: {e}", exc_info=True)
            raise

        finally:
            if job.job_id in self._active_jobs:
                del self._active_jobs[job.job_id]

    def start_worker(self, prefetch_count: int = 1):
        """
        Start processing jobs from the queue.

        Args:
            prefetch_count: Number of jobs to prefetch
        """
        if not self.connection:
            self.connect()

        logger.info(f"Starting worker for queue: {self.queue_name}")

        def on_message(ch, method, properties, body):
            try:
                job_data = json.loads(body)
                job = Job.from_dict(job_data)

                logger.info(f"Processing job {job.job_id} ({job.job_type})")

                try:
                    result = self._process_job(job)
                    logger.info(f"Job {job.job_id} completed successfully")
                    ch.basic_ack(delivery_tag=method.delivery_tag)

                except Exception as e:
                    # Retry logic
                    if job.retry_count < job.max_retries:
                        job.retry_count += 1
                        job.status = JobStatus.RETRYING
                        logger.warning(
                            f"Job {job.job_id} failed (attempt {job.retry_count}/{job.max_retries}), retrying..."
                        )

                        # Requeue with exponential backoff
                        time.sleep(2 ** job.retry_count)

                        ch.basic_publish(
                            exchange=self.exchange,
                            routing_key=method.routing_key,
                            body=json.dumps(job.to_dict()),
                            properties=properties
                        )
                        ch.basic_ack(delivery_tag=method.delivery_tag)

                    else:
                        logger.error(
                            f"Job {job.job_id} failed after {job.max_retries} retries, sending to DLQ"
                        )
                        # NACK and send to DLQ
                        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

        self.channel.basic_qos(prefetch_count=prefetch_count)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=on_message
        )

        try:
            logger.info("Worker started, waiting for jobs...")
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            self.channel.stop_consuming()
        finally:
            self.close()

    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Queue stats including message counts
        """
        if not self.connection:
            raise RuntimeError("Not connected to RabbitMQ")

        queue_info = self.channel.queue_declare(
            queue=self.queue_name,
            durable=True,
            passive=True
        )

        dlq_info = self.channel.queue_declare(
            queue=f"{self.queue_name}.dlq",
            durable=True,
            passive=True
        )

        return {
            "queue_name": self.queue_name,
            "pending_jobs": queue_info.method.message_count,
            "dead_letter_jobs": dlq_info.method.message_count,
            "active_jobs": len(self._active_jobs),
            "active_job_ids": list(self._active_jobs.keys())
        }
