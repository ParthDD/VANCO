from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "password123"

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

def check_connection():
    with driver.session() as session:
        result = session.run("RETURN 1 AS number")
        number = result.single()["number"]
        return number

if __name__ == "__main__":
    try:
        num = check_connection()
        print("Connected to Neo4j. 1 + 0 =", num)
    except Exception as e:
        print("Connection failed:", e)