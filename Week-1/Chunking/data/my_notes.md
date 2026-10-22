# Learning Journey Log

## Lesson 1 and 2: API basics and first LLM calls
An LLM call is a POST request. The prompt goes in the request body as JSON and the answer comes back as nested JSON. To read the text I dig down with data["candidates"][0]["content"]["parts"][0]["text"]. The API key is stored in Colab Secrets in Google Colab, and in a .env file in VS Code. The .env file must be listed in .gitignore before the first commit, because a committed key stays in the Git history even after the file is deleted.

## Lesson 3: Structured JSON output
Plain text is for people and structured JSON is for programs. A response schema forces fixed field names and types, so the code never gets a renamed field. json.loads converts a string into a Python dictionary, and json.dumps converts a dictionary into a string. If a field name is wrong, Python raises a KeyError. I built a skill-gap checker that extracts required skills from a job description and compares them to my own skills.

## Problems I hit and how I fixed them
Rate limit: my script failed with the error "Still rate limited after retries" because the free tier returned status 429. I changed the retry loop to wait 15, 30, 45 and 60 seconds, and I print the last response so I can see which limit was hit.

Wrong matches: exact matching said I was missing "python basics" even though I know Python. Substring matching fixed that but wrongly matched "java" inside "javascript". Matching whole words with a regular expression and the word boundary marker fixed both cases.

Where I defined has_skill: I had defined the function inside the loop. It worked, but it is cleaner to define it once above the loop.

## Lesson 4: Embeddings
An embedding turns text into a vector of 384 numbers using the all-MiniLM-L6-v2 model. Similar meanings get nearby vectors. My first test scored "REST API" against "calling web services" at only 0.26, because short phrases give noisy scores. Adding descriptions to each skill gives the model more context. No single threshold separated the right matches from the wrong ones: a wrong match scored 0.49 while correct matches scored 0.36 and 0.37.

## Lesson 5: Vector search
Documents are embedded once and the query is embedded at search time. With normalized vectors, a dot product equals cosine similarity. The value k is how many top results to return after sorting. My three test questions scored between 0.51 and 0.61 for the correct document. An irrelevant question about the capital of India scored only 0.08, so I added a minimum score so the app can say that no relevant documents were found. I also replaced hardcoded test queries with input() so the program accepts any question.

## Lesson 6: Chunking and Chroma
Long documents are cut into chunks of about 500 characters with a 100 character overlap. The chunks are stored in a Chroma database in a folder named chroma_db, which is listed in .gitignore. The database must be deleted and rebuilt after changing the chunk size or the embedding model.
