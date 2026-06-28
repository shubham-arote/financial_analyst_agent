"""Service layer — orchestration between the thin FastAPI routes and the core packages.

  * documents — document lifecycle (create, parse, persist, reload); owns DOCS + the DocStore
  * chat      — ask orchestration (retriever caching + the agent engine)
"""
