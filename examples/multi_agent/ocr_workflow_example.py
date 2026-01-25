"""
Multi-Agent OCR Workflow Example

Demonstrates how to use the multi-agent collaboration system
for OCR processing workflows.
"""

import os
import sys
import logging
import json
from pathlib import Path

# Add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ocr.communication.rabbitmq_transport import RabbitMQTransport

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_simple_preprocessing():
    """Example 1: Simple image preprocessing."""
    logger.info("=== Example 1: Simple Preprocessing ===")

    transport = RabbitMQTransport(
        host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
        agent_id="client.example1"
    )

    try:
        transport.connect()

        # Send preprocessing command
        response = transport.send_command(
            target="agent.ocr.preprocessor",
            command="normalize_image",
            payload={
                "image_path": "/data/sample_images/receipt_001.jpg",
                "options": {
                    "enhance_contrast": True,
                    "denoise": True
                }
            },
            timeout=30
        )

        result = response.get("payload", {})
        logger.info(f"Preprocessing result: {json.dumps(result, indent=2)}")

        if result.get("status") == "success":
            logger.info(f"✓ Image preprocessed successfully: {result.get('output_path')}")
        else:
            logger.error(f"✗ Preprocessing failed: {result.get('message')}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        transport.close()


def example_2_full_ocr_workflow():
    """Example 2: Complete OCR workflow via orchestrator."""
    logger.info("=== Example 2: Full OCR Workflow ===")

    transport = RabbitMQTransport(
        host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
        agent_id="client.example2"
    )

    try:
        transport.connect()

        # Execute full workflow
        response = transport.send_command(
            target="agent.orchestrator",
            command="execute_ocr_workflow",
            payload={
                "image_paths": [
                    "/data/sample_images/receipt_001.jpg",
                    "/data/sample_images/receipt_002.jpg"
                ],
                "workflow_config": {
                    "stages": ["preprocess", "inference", "validation"],
                    "preprocess_options": {
                        "enhance_contrast": True
                    },
                    "validation_options": {
                        "use_llm": True,
                        "min_confidence": 0.7
                    }
                },
                "output_dir": "/data/output/workflow_results"
            },
            timeout=300
        )

        result = response.get("payload", {})
        logger.info(f"Workflow result: {json.dumps(result, indent=2)}")

        if result.get("status") == "success":
            summary = result.get("summary", {})
            logger.info(f"✓ Workflow completed successfully")
            logger.info(f"  - Processed: {summary.get('successful')}/{summary.get('total_images')} images")
            logger.info(f"  - Total time: {summary.get('total_time_seconds', 0):.2f}s")
            logger.info(f"  - Avg quality: {summary.get('avg_quality_score', 0):.2f}")
        else:
            logger.error(f"✗ Workflow failed: {result.get('message')}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        transport.close()


def example_3_llm_powered_validation():
    """Example 3: LLM-powered OCR validation."""
    logger.info("=== Example 3: LLM-Powered Validation ===")

    transport = RabbitMQTransport(
        host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
        agent_id="client.example3"
    )

    try:
        transport.connect()

        # Mock OCR result for validation
        mock_ocr_result = {
            "results": [
                {"text": "Receipt #12345", "confidence": 0.95, "bbox": [10, 10, 200, 30]},
                {"text": "Total: $45.99", "confidence": 0.88, "bbox": [10, 100, 200, 120]},
                {"text": "Date: 01/15/2024", "confidence": 0.92, "bbox": [10, 50, 200, 70]}
            ],
            "full_text": "Receipt #12345 Date: 01/15/2024 Total: $45.99"
        }

        # Validate with LLM
        response = transport.send_command(
            target="agent.ocr.validator",
            command="validate_ocr_result",
            payload={
                "ocr_result": mock_ocr_result,
                "validation_rules": {
                    "min_confidence": 0.7,
                    "min_quality_score": 0.8
                },
                "use_llm": True
            },
            timeout=60
        )

        result = response.get("payload", {})
        logger.info(f"Validation result: {json.dumps(result, indent=2)}")

        if result.get("status") == "success":
            logger.info(f"✓ Validation completed")
            logger.info(f"  - Valid: {result.get('is_valid')}")
            logger.info(f"  - Quality score: {result.get('quality_score', 0):.2f}")
            logger.info(f"  - Issues found: {len(result.get('issues', []))}")

            if result.get("suggestions"):
                logger.info("  - Suggestions:")
                for suggestion in result.get("suggestions", []):
                    logger.info(f"    * {suggestion}")
        else:
            logger.error(f"✗ Validation failed: {result.get('message')}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        transport.close()


def example_4_batch_processing():
    """Example 4: Batch OCR processing using job queue."""
    logger.info("=== Example 4: Batch Processing ===")

    from ocr.workers.job_queue import JobQueue, JobPriority

    queue = JobQueue(
        queue_name="jobs.ocr",
        rabbitmq_host=os.getenv("RABBITMQ_HOST", "rabbitmq")
    )

    try:
        queue.connect()

        # Submit batch job
        job_id = queue.submit_job(
            job_type="ocr.batch",
            payload={
                "image_dir": "/data/sample_images",
                "output_dir": "/data/output/batch_results",
                "batch_size": 10,
                "workflow_config": {
                    "stages": ["preprocess", "inference"],
                    "preprocess_options": {
                        "enhance_contrast": True
                    }
                }
            },
            priority=JobPriority.HIGH,
            max_retries=2,
            timeout=1800  # 30 minutes
        )

        logger.info(f"✓ Batch job submitted: {job_id}")
        logger.info("  Job will be processed by background workers")

        # Check queue stats
        stats = queue.get_queue_stats()
        logger.info(f"Queue stats: {json.dumps(stats, indent=2)}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        queue.close()


def example_5_workflow_planning():
    """Example 5: AI-powered workflow planning."""
    logger.info("=== Example 5: AI-Powered Workflow Planning ===")

    transport = RabbitMQTransport(
        host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
        agent_id="client.example5"
    )

    try:
        transport.connect()

        # Request workflow planning
        response = transport.send_command(
            target="agent.orchestrator",
            command="plan_workflow",
            payload={
                "requirements": """
                I need to process 1000 receipt images.
                Requirements:
                - Extract text with high accuracy
                - Validate extracted amounts and dates
                - Must complete within 2 hours
                - Need to prioritize quality over speed
                """,
                "constraints": {
                    "max_time_hours": 2,
                    "min_quality": 0.9,
                    "available_workers": 2
                }
            },
            timeout=30
        )

        result = response.get("payload", {})
        logger.info(f"Workflow plan: {json.dumps(result, indent=2)}")

        if result.get("status") == "success":
            plan = result.get("workflow_plan", {})
            logger.info(f"✓ Workflow plan created")
            logger.info(f"  - Stages: {plan.get('stages', [])}")
            logger.info(f"  - Estimated time/image: {plan.get('estimated_time_per_image', 0):.2f}s")
            logger.info(f"  - Reasoning: {plan.get('reasoning', 'N/A')}")
        else:
            logger.error(f"✗ Planning failed: {result.get('message')}")

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        transport.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Multi-Agent OCR Workflow Examples")
    print("=" * 60 + "\n")

    examples = [
        ("Simple Preprocessing", example_1_simple_preprocessing),
        ("Full OCR Workflow", example_2_full_ocr_workflow),
        ("LLM-Powered Validation", example_3_llm_powered_validation),
        ("Batch Processing", example_4_batch_processing),
        ("AI-Powered Planning", example_5_workflow_planning)
    ]

    for i, (name, func) in enumerate(examples, 1):
        print(f"\nRunning Example {i}: {name}")
        print("-" * 60)
        try:
            func()
        except Exception as e:
            logger.error(f"Example {i} failed: {e}", exc_info=True)
        print()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
