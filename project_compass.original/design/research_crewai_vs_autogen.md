# Research: CrewAI vs AutoGen for Multi-Agent Infrastructure

**Date:** 2026-01-12
**Status:** Completed
**Recommendation:** **AutoGen**

## Executive Summary
For the purpose of building a "Multi-Agent Collaboration Environment" with a custom "Inter-Agent Communication Protocol" (IACP) and RabbitMQ integration, **AutoGen** is the recommended framework. Its lower-level, message-driven architecture (Actor Model) provides the necessary primitives to implement custom wire protocols and integrate with external message brokers more naturally than CrewAI's higher-level orchestrator pattern.

## Detailed Comparison

| Feature                  | CrewAI                                                                                                                | AutoGen                                                                                                                           |
| :----------------------- | :-------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**         | **Orchestrator-Workers**: Hierarchical, role-based. Best for linear or DAG-based processes.                           | **Actor Model**: Distributed, conversational. Agents are "actors" that send/receive messages.                                     |
| **Custom Protocols**     | **Limited**: Uses internal "Agent-to-Agent" (JSON-RPC) protocol. optimizing for role delegation.                      | **High**: Allows subclassing `BaseChatMessage`. "Custom Brokers" can be implemented to route messages via arbitrary transports.   |
| **RabbitMQ Integration** | **External**: Can trigger Crews via Celery/Redis, but internal agent-to-agent chat is tightly coupled to the library. | **Native-aligned**: The event-driven loop allows replacing the transport layer. Community libraries exist for RabbitMQ streaming. |
| **Background Agents**    | **Supported**: Has `kickoff_async` and async tasks. Good for "fire and forget".                                       | **Excellent**: Agents can run in separate processes/containers and communicate solely via events (e.g., watching a queue).        |
| **Strict Typing**        | **Strong**: Heavy reliance on Pydantic for task outputs.                                                              | **Strong**: Uses Pydantic for agent interfaces (`BaseModel`) and tool schemas. Best suited for rigid contract enforcement.        |

## Key Findings

### 1. Custom Protocol Support (IACP)
*   **CrewAI** is opinionated about how agents talk (delegation, hierarchy). Injecting a custom IACP (e.g., "Agent A *must* send Schema X to Agent B via Queue Y") is fighting the framework.
*   **AutoGen** treats "conversation" as the universal interface. We can define a "conversation" that is actually a structured data exchange over RabbitMQ, enforcing compliance via the IACP schema.

### 2. Infrastructure Suitability
*   **CrewAI** is excellent for *applications* (e.g., "Write a blog post").
*   **AutoGen** is better for *infrastructure* (e.g., "Maintain a fleet of coding agents that listen for linting jobs"). Key difference: AutoGen agents can persist and react to events; CrewAI crews are typically ephemeral (run once and die).

### 3. Observability
*   AutoGen's explicit message passing makes it easier to "tap" the wire (RabbitMQ) to log, replay, or audit every interaction without modifying the agent code.

## Recommendation Strategy

Proceed with **AutoGen** as the core framework for the "Multi-Agent Collaboration Environment".

1.  **Core Abstraction**: Wrap AutoGen agents with a "Transport Layer" that speaks IACP over RabbitMQ.
2.  **Hybrid Approach**: We can still use simplistic CrewAI-like patterns *within* a single node if needed, but the *inter-node* communication should be AutoGen + RabbitMQ.
3.  **Protocol**: Define IACP using Pydantic, which both frameworks support, but AutoGen will consume more natively.
