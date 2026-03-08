from cache.query_engine import QueryEngine

engine = QueryEngine()

while True:

    q = input("\nEnter query: ")

    response = engine.query(q)

    print("\nResponse:\n")

    for k, v in response.items():
        print(k, ":", v)

    print("\nCache Stats:")
    print(engine.cache.stats())