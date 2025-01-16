import wikipedia
import json
import time
import asyncio
import aiohttp
from functools import lru_cache

@lru_cache(maxsize=2000)
def search_wikipedia(subtopic, results=1000):
    return wikipedia.search(subtopic, results=results)

async def fetch_revision_id(session, result):
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={result.replace(' ', '_')}&prop=revisions&rvprop=ids&format=json"
        async with session.get(url) as response:
            data = await response.json()
            page_id = list(data["query"]["pages"].keys())[0]
            revision_id = data["query"]["pages"][page_id]["revisions"][0]["revid"]
            await asyncio.sleep(1)  
            return revision_id
    except Exception as e:
        pass
        return None

async def fetch_summary(session, result, topic, subtopic):
    try:
        url = f'https://en.wikipedia.org/api/rest_v1/page/summary/{result.replace(" ", "_")}'
        async with session.get(url) as response:
            data = await response.json()
            summary = data.get("extract", "")
            if len(summary) > 300:  
                
                revision_id = await fetch_revision_id(session, result)
                await asyncio.sleep(1)  
                return {
                    "title": result,
                    "summary": summary,
                    "url": f'https://en.wikipedia.org/wiki/{result.replace(" ", "_")}',
                    "topic": topic,
                    "subtopic": subtopic,
                    "revision_id": revision_id  
                }
    except Exception as e:
        print(f"Error fetching summary for {result}: {e}")
        return None

async def scrape_subtopic(subtopic, topic, seen_titles):
    search_results = search_wikipedia(subtopic)
    unique_docs = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for result in search_results:
            if result not in seen_titles:  
                seen_titles.add(result)
                tasks.append(fetch_summary(session, result, topic, subtopic))

        summaries = await asyncio.gather(*tasks)
        unique_docs = [summary for summary in summaries if summary is not None]
    return unique_docs

async def scrape_topic(topic, subtopics, seen_titles):
    all_docs = []
    for subtopic in subtopics:
        print(f"Scraping subtopic: {subtopic}")
        docs = await scrape_subtopic(subtopic, topic, seen_titles)
        all_docs.extend(docs)
    return all_docs

async def main():
    topics = {
        "Health": ["Common diseases", "Global health statistics", "Mental health trends"],
        "Environment": ["Global warming", "Endangered species", "Deforestation rates"],
        "Technology": ["Emerging technologies", "AI advancements"],
        "Economy": ["Stock market performance", "Job markets", "Cryptocurrency trends"],
        "Entertainment": ["Music industry", "Cultural events", "Streaming platforms"],
        "Sports": ["Major sporting events", "Sports analytics"],
        "Politics": ["Elections", "Public policy analysis", "International relations"],
        "Education": ["Literacy rates", "Online education trends", "Student loan data"],
        "Travel": ["Top tourist destinations", "Airline industry data", "Travel trends"],
        "Food": ["Crop yield statistics", "Global hunger", "Food security"],
    }

    all_data = {}
    seen_titles = set() 
    total_docs = 0  

    for topic, subtopics in topics.items():
        print(f"Scraping topic: {topic}")
        documents = await scrape_topic(topic, subtopics, seen_titles)
        all_data[topic] = documents
        total_docs += len(documents)
        print(f"Retrieved {len(documents)} unique documents for {topic}")

    with open("2uj_data.json", "w") as f:
        json.dump(all_data, f, indent=4)

    print(f"Total unique documents retrieved across all topics: {total_docs}")

asyncio.run(main())
