# FastAPI Notes

What I learned about building an API with FastAPI:

- `@app.get` reads a value and returns it from the API.
- `@app.post` creates new data, validated by a Pydantic model.
- `@app.put` updates data that already exists.
- `@app.delete` removes a value from the database.

Run the dev server with `uvicorn main:app --reload`.
