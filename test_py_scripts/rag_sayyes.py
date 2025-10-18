from __future__ import annotations
import httpx
from typing import Optional, Dict, Any, Callable, Awaitable, Set, List 
import faiss
import numpy as np
import asyncio
import json
from bson import ObjectId
from openai import AsyncOpenAI
from urllib.parse import urlparse
import pandas as pd
import json
import asyncio
import logging
import os, json, logging, uuid, requests, boto3 
import os
import re
import logging, os, uuid
from urllib.parse import urlsplit, urlunsplit
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from openai import OpenAI
from pymongo import MongoClient
import time
import psycopg2
from dotenv import load_dotenv
load_dotenv() 

current_dir = os.getcwd()

MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")

client = MongoClient(MONGO_CONNECTION_STRING)
db = client.get_database()
# from prompts import PROMPT

S3_BUCKET_NAME = "sayyes-image-storage"
AWS_REGION = "us-east-1"                      # ← change if needed

s3 = boto3.client("s3", region_name=AWS_REGION)
# log = logging.getLogger("rehost")
# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")



OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)



# -----------------------------------------------------------------------------#
#  BaseVectorizer Class
# -----------------------------------------------------------------------------#

_perplexity_client_pool = None
_pool_lock = asyncio.Lock()

async def get_perplexity_client():
    """Get or create the shared HTTP client."""
    global _perplexity_client_pool
    
    async with _pool_lock:
        if _perplexity_client_pool is None:
            _perplexity_client_pool = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=20.0, read=60.0),
                limits=httpx.Limits(
                    max_connections=50,  # Increased
                    max_keepalive_connections=30  # Increased
                ),
                http2=True,
                follow_redirects=True
            )
            print("✓ Created persistent Perplexity HTTP client")
    
    return _perplexity_client_pool

class OptimizedPerplexityVendorSearch:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.perplexity.ai/chat/completions"
        # Create persistent connection pool for reuse
        self.client = None
        
    async def __aenter__(self):
        if not self.client:
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=20.0, read=60.0),  # Increased timeouts
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
                http2=True
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    def _build_optimized_prompt(self, user_query: str, location: str, category: str, disliked_titles: List[str]) -> str:
        """Shorter, more focused prompt for faster processing.

        Adds a strict constraint to avoid recommending vendors with zero or fewer reviews.
        """
        disliked_text = f" Exclude: {', '.join(disliked_titles)}" if disliked_titles else ""
        
        return f"""Find 8 top {category} vendors for weddings in {location} matching: {user_query}{disliked_text}

    Return ONLY this JSON array:
    [{{"name":"Business Name","address":"Full Address","location":"City, State","contact_number":"Phone","website_url":"Website","vendor_rating":"4.5","reviews_count":"100","about":"Brief description","pricing":"Price range","service":"Services offered","specialties":"What makes them special","capacity":"Guest capacity","availability":"General availability info","style":"Style specialization","notes":"Additional details or awards"}}]

    Requirements: Located in/serves {location}, currently operational, good ratings preferred.
    STRICT: Do not include vendors with reviews_count <= 0 or missing/unknown. Reviews must be greater than 5.
    JSON only, no explanations."""

    def _extract_json_fast(self, text: str) -> Optional[List[Dict]]:
        """Faster JSON extraction with better error handling."""
        text = text.strip()
        
        # Direct JSON parse first (fastest)
        if text.startswith('[') and text.endswith(']'):
            try:
                return json.loads(text)
            except:
                pass
        
        # Extract between brackets
        try:
            start = text.find('[')
            end = text.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except:
            pass
        
        # Code block extraction
        code_patterns = [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']
        for pattern in code_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    continue
        return None

    async def search_vendors_optimized(
        self, 
        user_query: str, 
        location: str, 
        category: str,
        top_n: int = 8,
        disliked_titles: List[str] = []
    ) -> List[Dict[str, Any]]:
        """Optimized vendor search with multiple performance improvements."""
        
        prompt = self._build_optimized_prompt(user_query, location, category, disliked_titles)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Optimized API request body
        body = {
            "model": "sonar-pro",  # Use faster model instead of sonar-pro
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            # "max_tokens": 8000,# Limit response size
            "top_p": 0.9
        }
        
        try:
            response = await self.client.post(
                self.api_url,
                headers=headers,
                json=body
            )
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            if response.status_code != 200:
                response_text = response.text
                print(f"Error response body: {response_text}")
                return []
            response.raise_for_status()
            data = response.json()

            print(f"Response data structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
      
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            vendors = self._extract_json_fast(content)
            
            if not vendors:
                logging.warning(f"No vendor data parsed from Perplexity response")
                return []
            
            # Quick format and validate
            formatted_vendors = []
            for vendor in vendors[:top_n]:
                if isinstance(vendor, dict) and vendor.get("name"):
                    formatted_vendor = self._format_vendor_fast(vendor)
                    formatted_vendors.append(formatted_vendor)
            
            return formatted_vendors
            
        except httpx.HTTPStatusError as e:
            print(f"HTTP Status Error: {e.response.status_code}")
            print(f"Error response: {e.response.text}")
            logging.error(f"HTTP error {e.response.status_code}: {e.response.text}")
            return []
        except httpx.TimeoutException:
            print("Request timed out")
            logging.error("Perplexity API timeout")
            return []
        except Exception as e:
            print(f"Unexpected error: {type(e).__name__}: {str(e)}")
            logging.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
            return []

    def _format_vendor_fast(self, vendor: Dict) -> Dict[str, Any]:
        """Faster vendor formatting with minimal processing."""
        return {
            "title": vendor.get("name", ""),
            "address": vendor.get("address", vendor.get("location", "")),
            "contact_number": vendor.get("contact_number", ""),
            "website_url": self._clean_url_fast(vendor.get("website_url", "")),
            "vendor_rating": vendor.get("vendor_rating"),
            "reviews_count": vendor.get("reviews_count"),
            "about": vendor.get("about", ""),
            "pricing": vendor.get("pricing", ""),
            "details": {
                "service": vendor.get("service", ""),
                "specialties": vendor.get("specialties", ""),
                "capacity": vendor.get("capacity", ""),
                "style": vendor.get("style", "")
            },
            "image_urls": [],  # Will be filled by Google Places API
            "reviews": []      # Will be filled by Google Places API
        }

    def _clean_url_fast(self, url: str) -> str:
        """Minimal URL cleaning for speed."""
        if not url or not isinstance(url, str):
            return ""
        url = url.strip()
        if url and not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url.split('?')[0]


# 2. REPLACE the get_vendor_recommendations_with_details_perplexity function (lines 184-209) with:

async def get_vendor_recommendations_with_details_perplexity(
    user_query: str, 
    category: str, 
    location: str, 
    disliked_titles: List[str]
) -> List[Dict]:
    """Optimized version with connection pooling and faster processing."""
    
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    print(f"User Query: {user_query}, Location: {location}, Category: {category}")
    
    async with OptimizedPerplexityVendorSearch(PERPLEXITY_API_KEY) as searcher:
        try:
            start_time = time.time()
            
            # Get vendors from Perplexity (should be much faster now)
            vendors = await searcher.search_vendors_optimized(
                user_query=user_query,
                location=location,
                category=category,
                top_n=8,
                disliked_titles=disliked_titles
            )
            
            perplexity_time = time.time()
            print(f"Perplexity API call took {perplexity_time - start_time:.2f} seconds")
            
            # Enrich with Google data concurrently (if needed)
            if vendors:
                enriched_vendors = await enrich_vendors_concurrent(vendors)
                
                end_time = time.time()
                print(f"Total processing took {end_time - start_time:.2f} seconds")
                print(f"Google Places enrichment took {end_time - perplexity_time:.2f} seconds")
                
                return enriched_vendors
            
            return vendors
            
        except Exception as e:
            logging.error(f"Error in optimized vendor search: {str(e)}")
            return []



async def enrich_vendors_concurrent(vendors: List[Dict]) -> List[Dict]:
    """Enrich vendors with Google data using concurrent requests."""
    
    async def enrich_single_vendor(vendor):
        try:
            place_id = await get_place_id_async(vendor["title"], vendor["address"])
            if place_id:
                image_urls, reviews = await get_images_and_reviews_async(place_id)
                vendor["image_urls"] = image_urls
                vendor["reviews"] = reviews
            return vendor
        except Exception as e:
            logging.error(f"Error enriching vendor {vendor.get('title', '')}: {e}")
            vendor["image_urls"] = []
            vendor["reviews"] = []
            return vendor
    
    # Process all vendors concurrently
    tasks = [enrich_single_vendor(vendor) for vendor in vendors]
    enriched_vendors = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter out exceptions and return valid results
    return [v for v in enriched_vendors if isinstance(v, dict)]


async def get_place_id_async(name: str, address: str) -> Optional[str]:
    """Async version of get_place_id"""
    search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{name} {address}",
        "key": GOOGLE_API_KEY
    }
    
    timeout = httpx.Timeout(10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(search_url, params=params)
            resp_data = response.json()
            if resp_data.get("results"):
                return resp_data["results"][0]["place_id"]
    except Exception as e:
        logging.error(f"Error getting place_id for {name}: {e}")
    
    return None


async def get_images_and_reviews_async(place_id: str) -> tuple:
    """Async version of get_images_and_reviews"""
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "photos,reviews"
    }
    
    timeout = httpx.Timeout(10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(details_url, params=params)
            resp_data = response.json()
            result = resp_data.get("result", {})

            # Images
            image_urls = []
            for photo in result.get("photos", []):
                ref = photo.get("photo_reference")
                if ref:
                    from urllib.parse import urlencode
                    url = f"https://maps.googleapis.com/maps/api/place/photo?{urlencode({'maxwidth': 1200, 'photoreference': ref, 'key': GOOGLE_API_KEY})}"
                    image_urls.append(url)

            # Reviews
            reviews = json.dumps([
                {
                    "author_name": rev.get("author"),
                    "rating": rev.get("rating"),
                    "text": rev.get("text"),
                    "relative_time": rev.get("time")
                }
                for rev in result.get("reviews", [])
            ])

            return image_urls, reviews
            
    except Exception as e:
        logging.error(f"Error getting images/reviews for place_id {place_id}: {e}")
    
    return [], []



# -----------------------------------------------------------------------------#
#  Google Place Details API after perplexity
# -----------------------------------------------------------------------------#



GOOGLE_API_KEY = "AIzaSyB6n5d427zt4w9WKvknyBZOPQGIrx8gAqY"

from urllib.parse import urlencode

def get_place_id(name, address):
    """Get Google place_id from name + address using Text Search API."""
    search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{name} {address}",
        "key": GOOGLE_API_KEY
    }
    resp = requests.get(search_url, params=params).json()
    if resp.get("results"):
        return resp["results"][0]["place_id"]
    return None

def get_images_and_reviews(place_id):
    """Fetch photos & reviews from Google Place Details API."""
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "photos,reviews"
    }
    resp = requests.get(details_url, params=params).json()
    result = resp.get("result", {})

    # Images
    image_urls = []
    for photo in result.get("photos", []):
        ref = photo.get("photo_reference")
        if ref:
            url = f"https://maps.googleapis.com/maps/api/place/photo?{urlencode({'maxwidth': 1200, 'photoreference': ref, 'key': GOOGLE_API_KEY})}"
            image_urls.append(url)

    # Reviews
    reviews = json.dumps([
        {
            "author_name": rev.get("author"),
            "rating": rev.get("rating"),
            "text": rev.get("text"),
            "relative_time": rev.get("time")
        }
        for rev in result.get("reviews", [])
    ])

    return image_urls, reviews

def enrich_vendors_with_google_data(vendor_list):
    """Add image URLs & reviews to each vendor in the given list."""
    for vendor in vendor_list:
        place_id = get_place_id(vendor["title"], vendor["address"])
        if place_id:
            image_urls, reviews = get_images_and_reviews(place_id)
            vendor["image_urls"] = image_urls
            vendor["reviews"] = reviews
        else:
            vendor["image_urls"] = []
            vendor["reviews"] = []
    print(f"Enriched vendors with Google data: {len(vendor_list)}")
    return vendor_list



class BaseVectorizer:
    def __init__(
        self,
        product_table: str,
        csvFilePath,
        FAISS_INDEX_PATH: str,
        id_INDEX_STORAGE_PATH: str,
        embedding_model,
        entity_name: str,
        field_map: Dict[str, str],
        details_fields: list[str],
        extra_fields: list[str] = None,
    ):
        self.product_table = product_table
        self.csvFilePath = csvFilePath
        self.entity_name = entity_name  # e.g. "dj", "florist"
        self.field_map = field_map      # maps output keys to df columns
        self.details_fields = details_fields
        self.extra_fields = extra_fields or []
        self.embedding_model = embedding_model

        # Load FAISS index and ID index if paths are given
        self.faiss_index_slot = faiss.read_index(FAISS_INDEX_PATH) if os.path.exists(FAISS_INDEX_PATH) else None
        self.id_to_index_slot = np.load(id_INDEX_STORAGE_PATH, allow_pickle=True).item() if os.path.exists(id_INDEX_STORAGE_PATH) else None

    def generate_embeddings(self, text_list: list[str]):
        return self.embedding_model.embed_documents(text_list)

    def generate_embedding(self, text: str):
        return self.embedding_model.embed_query(text)
    
    def get_metadata_by_id(self , table : str  , record_id : int) -> pd.DataFrame:

        DATABASE_URL = "postgresql://postgres.mdlcfczkzpmhblazstpm:hemanthterli@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table} WHERE id = %s;", (record_id,))
        row = cursor.fetchone()
        cols = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        return pd.DataFrame([row], columns=cols)

    def _process_disliked_titles(self, disliked_titles):
        import json
        if disliked_titles and isinstance(disliked_titles, str):
            try:
                disliked_titles = json.loads(disliked_titles)
            except Exception:
                disliked_titles = []
        return disliked_titles or []

    def _get_base_url(self, website_url: str) -> str:
        if isinstance(website_url, str) and website_url.strip():
            try:
                parsed_url = urlparse(website_url)
                if parsed_url.scheme and parsed_url.netloc:
                    return f"{parsed_url.scheme}://{parsed_url.netloc}"
                elif parsed_url.netloc:
                    return f"https://{parsed_url.netloc}"
                elif parsed_url.path:
                    return f"https://{parsed_url.path.split('/')[0]}"
            except Exception:
                return website_url
        return ""

    def retrieve_similar(self, input_category, top_n=3, disliked_titles: Optional[list]=None , location: Optional[str]=None) -> list[Dict[str, Any]]:

        # Vectorize input
        start_vectorization_time = time.time()
        if isinstance(input_category, list):
            input_category = "|".join(input_category)
        input_vector = np.array(self.generate_embeddings([input_category])).reshape(1, -1)

        end_vectorization_time = time.time()

        ##update_run_time_log(f"Vectorization took {end_vectorization_time - start_vectorization_time} seconds for input: {input_category}")


        if location:
            location_vector = np.array(self.generate_embeddings([location])).reshape(1, -1)
            input_vector = location_vector + input_vector

        ##update_run_time_log(f"Input vector after location adjustment: {input_vector}")


        disliked_titles = self._process_disliked_titles(disliked_titles)
        if disliked_titles:
            neg_vectors = [self.generate_embeddings([title])[0] for title in disliked_titles]
            neg_mean = np.mean(neg_vectors, axis=0).reshape(1, -1)
            input_vector = input_vector - 0.8 * neg_mean

        # distances, indices = self.faiss_index_slot.search(input_vector, top_n)
        def search_similar_vendors(input_vector: np.ndarray, table_name , top_n: int = 5):
            start_time = time.time()

            # Flatten and format vector
            vector_array = np.array(input_vector).flatten()
            vector_str = f"[{','.join(map(str, vector_array))}]"

            # Vector similarity query (return full rows + distance)
            query = f"""
                SELECT *, embedding <#> '{vector_str}' AS distance
                FROM {table_name}
                ORDER BY embedding <#> '{vector_str}'
                LIMIT {top_n};
            """


            # Connect and execute
            conn = psycopg2.connect("postgresql://postgres.ocorqentdblehpclpbzu:sayyesaisupabase@aws-0-us-east-1.pooler.supabase.com:5432/postgres")
            cur = conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
        
            column_names = [desc[0] for desc in cur.description]

            cur.close()
            conn.close()

            # Convert to DataFrame with all columns + distance
            df = pd.DataFrame(rows, columns=column_names)

            elapsed = time.time() - start_time
            print(f"🔍 Retrieved {len(df)}  vendors in {elapsed:.4f} seconds")
            # print(df)
            return df
        
        
        
        start_fetching_metadata = time.time()
        # df = pd.read_csv(self.csvFilePath)
        base_name = os.path.splitext(os.path.basename(self.csvFilePath))[0]
        root_name = base_name.replace("_rows", "").lower()
        table_name = f"{root_name}_vendor_data"

        df = search_similar_vendors(input_vector, table_name , top_n)

        ##update_run_time_log(f"Fetching metadata took {time.time() - start_fetching_metadata} seconds for input: {input_category}")
        results = []
        for index, row in df.iterrows():
        # for entity_id in similar_ids:
        #     row = df[df['id'] == entity_id]
            # row = self.get_metadata_by_id(metadata_table_name, entity_id)
            # if row.empty:
            #     continue
            # entity = row.iloc[0]
            entity = row.to_dict()
            # Skip disliked
            if disliked_titles and entity[self.field_map['title']] in disliked_titles:
                continue
            # Format details
            details = {field: entity[field] for field in self.details_fields}

            result = {
                self.entity_name: {
                    "title": entity[self.field_map['title']],
                    "address": entity[self.field_map['address']],
                    "contact_number": entity[self.field_map['contact_number']],
                    "website_url": self._get_base_url(entity[self.field_map['website_url']]),
                    "vendor_rating": entity[self.field_map['vendor_rating']],
                    "reviews_count": entity[self.field_map['reviews_count']],
                    "about": entity[self.field_map['about']],
                    "pricing": entity[self.field_map['pricing']],
                    "details": details,
                    "image_urls": entity[self.field_map['image_urls']],
                    "reviews": entity[self.field_map['reviews']],
                }
            }
            # Add any extra fields
            for extra in self.extra_fields:
                result[self.entity_name][extra] = entity.get(extra)
            results.append(result)
        return results

# -----------------------------------------------------------------------------#

#  Specific Vectorizer Classes
startTime = time.time()

FAISS_INDEX_PATH_bar_services = r'indexes\faiss_index_Bar_Services_data.index'
id_INDEX_STORAGE_PATH_bar_services = r'indexes\id_index_Bar_Services_data.npy'
bar_services_filename = r'Bar_Services_rows.csv'
bar_services_file_path = os.path.join(current_dir, bar_services_filename)
# data_bar_services = pd.read_csv(bar_services_file_path)

bar_service_field_map = {
    "title": "bar_service_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

bar_service_details_fields = ["bars_drinks"]

vectorizer_bar_service = BaseVectorizer(
    product_table="bar_service_data",
    csvFilePath = bar_services_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_bar_services,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_bar_services,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="bar_service",
    field_map=bar_service_field_map,
    details_fields=bar_service_details_fields,
    extra_fields=["social_urls", "source_url"]
)



FAISS_INDEX_PATH_beauty = r'indexes\faiss_index_BeautyRows_data.index'
id_INDEX_STORAGE_PATH_beauty = r'indexes\id_index_BeautyRows_data.npy'
beauty_filename = r'Beautyrows.csv'
beauty_file_path = os.path.join(current_dir, beauty_filename)
# data_beauty = pd.read_csv(beauty_file_path)

beauty_field_map = {
    "title": "beauty_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

beauty_details_fields = ["beauty"]

vectorizer_beauty = BaseVectorizer(
    product_table="beauty_data",
    csvFilePath=beauty_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_beauty,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_beauty,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="beauty",
    field_map=beauty_field_map,
    details_fields=beauty_details_fields,
    extra_fields=["social_urls", "source_url"]
)



FAISS_INDEX_PATH_bridal_salon = r'indexes\faiss_index_BridalSalons_data.index'
id_INDEX_STORAGE_PATH_bridal_salon = r'indexes\id_index_BridalSalons_data.npy'
bridal_salons_filename = r'BridalSalons_rows.csv'
bridal_salons_file_path = os.path.join(current_dir, bridal_salons_filename)
# data_bridal_salon = pd.read_csv(bridal_salons_file_path)

bridal_salon_field_map = {
    "title": "bridal_salon_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

bridal_salon_details_fields = [
    "accessories",
    "dresses",
    "fashion_services",
    "suits_accessories"
]

vectorizer_bridal_salon = BaseVectorizer(
    product_table="bridal_salon_data",
    csvFilePath=bridal_salons_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_bridal_salon,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_bridal_salon,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="bridal_salon",
    field_map=bridal_salon_field_map,
    details_fields=bridal_salon_details_fields,
    extra_fields=["social_urls", "source_url"]
)



FAISS_INDEX_PATH_catering = r'indexes\faiss_index_Catering_data.index'
id_INDEX_STORAGE_PATH_catering = r'indexes\id_index_Catering_data.npy'
catering_filename = r'Catering_rows.csv'
catering_file_path = os.path.join(current_dir, catering_filename)
# data_catering = pd.read_csv(catering_file_path)


catering_field_map = {
    "title": "catering_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

catering_details_fields = [
    "cuisine",
    "dietary_options",
    "food_catering"
]

vectorizer_catering = BaseVectorizer(
    product_table="catering_data",
    csvFilePath=catering_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_catering,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_catering,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="catering",
    field_map=catering_field_map,
    details_fields=catering_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_decor = r'indexes\faiss_index_Decor_data.index'
id_INDEX_STORAGE_PATH_decor = r'indexes\id_index_Decor_data.npy'
decor_filename = r'Decor_rows.csv'
decor_file_path = os.path.join(current_dir, decor_filename)
# data_decor = pd.read_csv(decor_file_path)

decor_field_map = {
    "title": "decor_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

decor_details_fields = [
    "decorations_accents",
    "invitations_paper_goods",
    "lighting",
    "rentals_equipment"
]

vectorizer_decor = BaseVectorizer(
    product_table="decor_data",
    csvFilePath=decor_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_decor,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_decor,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="decor",
    field_map=decor_field_map,
    details_fields=decor_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_djs = r'indexes\faiss_index_DJs_data.index'
id_INDEX_STORAGE_PATH_djs = r'indexes\id_index_DJs_data.npy'
djs_filename = r'DJs_rows.csv'
djs_file_path = os.path.join(current_dir, djs_filename)
# data_djs = pd.read_csv(djs_file_path)

dj_field_map = {
    "title": "dj_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

dj_details_fields = [
    "equipment",
    "music_genres",
    "music_services",
    "wedding_activites"
]

vectorizer_dj = BaseVectorizer(
    product_table="dj_data",
    csvFilePath=djs_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_djs,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_djs,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="dj",
    field_map=dj_field_map,
    details_fields=dj_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_ensembles_soloists = r'indexes\faiss_index_Ensembles_Soloists_data.index'
id_INDEX_STORAGE_PATH_ensembles_soloists = r'indexes\id_index_Ensembles_Soloists_data.npy'
ensembles_soloists_filename = r'Ensembles_Soloists_rows.csv'
ensembles_soloists_file_path = os.path.join(current_dir, ensembles_soloists_filename)
# data_ensembles_soloists = pd.read_csv(ensembles_soloists_file_path)

ensembles_soloists_field_map = {
    "title": "ensembles_soloists_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

ensembles_soloists_details_fields = [
    "instruments",
    "music_genres",
    "music_services",
    "wedding_activites"
]

vectorizer_ensembles_soloists = BaseVectorizer(
    product_table="ensembles_soloists_data",
    csvFilePath=ensembles_soloists_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_ensembles_soloists,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_ensembles_soloists,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="ensembles_soloists",
    field_map=ensembles_soloists_field_map,
    details_fields=ensembles_soloists_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_florists = r'indexes\faiss_index_florists_data.index'
id_INDEX_STORAGE_PATH_florists = r'indexes\id_index_florists_data.npy'
florists_filename = r'Florists_rows.csv'
florists_file_path = os.path.join(current_dir, florists_filename)
# data_florists = pd.read_csv(florists_file_path)

florist_field_map = {
    "title": "florist_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

florist_details_fields = [
    "decorations_accents",
    "flower_arrangements"
]

vectorizer_florist = BaseVectorizer(
    product_table="florist_data",
    csvFilePath=florists_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_florists,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_florists,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="florist",
    field_map=florist_field_map,
    details_fields=florist_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_gifts_favors = r'indexes\faiss_index_Gifts_Favors_data.index'
id_INDEX_STORAGE_PATH_gifts_favors = r'indexes\id_index_Gifts_Favors_data.npy'
gifts_favors_filename = r'Gifts_Favors_rows.csv'
gifts_favors_file_path = os.path.join(current_dir, gifts_favors_filename)
# data_gifts_favors = pd.read_csv(gifts_favors_file_path)

gifts_favors_field_map = {
    "title": "gifts_favors_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

gifts_favors_details_fields = ["gifts_favors"]

vectorizer_gifts_favors = BaseVectorizer(
    product_table="gifts_favors_data",
    csvFilePath=gifts_favors_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_gifts_favors,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_gifts_favors,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="gifts_favors",
    field_map=gifts_favors_field_map,
    details_fields=gifts_favors_details_fields,
    extra_fields=["social_urls", "source_url"]
)



FAISS_INDEX_PATH_invitations_paper_goods = r'indexes\faiss_index_Invitations_Paper_Goods_data.index'
id_INDEX_STORAGE_PATH_invitations_paper_goods = r'indexes\id_index_Invitations_Paper_Goods_data.npy'
invitations_paper_goods_filename = r'Invitations_Paper_Goods_rows.csv'
invitations_paper_goods_file_path = os.path.join(current_dir, invitations_paper_goods_filename)
# data_invitations_paper_goods = pd.read_csv(invitations_paper_goods_file_path)


invitations_paper_goods_field_map = {
    "title": "invitations_paper_goods_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

invitations_paper_goods_details_fields = ["invitations_paper_goods"]

vectorizer_invitations_paper_goods = BaseVectorizer(
    product_table="invitations_paper_goods_data",
    csvFilePath=invitations_paper_goods_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_invitations_paper_goods,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_invitations_paper_goods,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="invitations_paper_goods",
    field_map=invitations_paper_goods_field_map,
    details_fields=invitations_paper_goods_details_fields,
    extra_fields=["social_urls", "source_url"]
)



FAISS_INDEX_PATH_jeweleres = r'indexes\faiss_index_Jeweleres_data.index'
id_INDEX_STORAGE_PATH_jeweleres = r'indexes\id_index_Jeweleres_data.npy'
jewelers_filename = r'Jeweleres_rows.csv'
jewelers_file_path = os.path.join(current_dir, jewelers_filename)
# data_jewelers = pd.read_csv(jewelers_file_path)

jeweler_field_map = {
    "title": "jeweler_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

jeweler_details_fields = ["rings", "wedding_jewelry"]

vectorizer_jeweler = BaseVectorizer(
    product_table="jeweler_data",
    csvFilePath=jewelers_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_jeweleres,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_jeweleres,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="jeweler",
    field_map=jeweler_field_map,
    details_fields=jeweler_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_photobooths = r'indexes\faiss_index_PhotoBooths_data.index'
id_INDEX_STORAGE_PATH_photobooths = r'indexes\id_index_PhotoBooths_data.npy'
photo_booths_filename = r'PhotoBooths_rows.csv'
photo_booths_file_path = os.path.join(current_dir, photo_booths_filename)
# data_photo_booths = pd.read_csv(photo_booths_file_path)

photo_booths_field_map = {
    "title": "photo_booths_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

photo_booths_details_fields = ["photo_video"]

vectorizer_photo_booths = BaseVectorizer(
    product_table="photo_booths_data",
    csvFilePath=photo_booths_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_photobooths,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_photobooths,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="photo_booths",
    field_map=photo_booths_field_map,
    details_fields=photo_booths_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_receptionvenues = r'indexes\faiss_index_ReceptionVenues_data.index'
id_INDEX_STORAGE_PATH_receptionvenues = r'indexes\id_index_ReceptionVenues_data.npy'
reception_venues_filename = r'ReceptionVenues_rows.csv'
reception_venues_file_path = os.path.join(current_dir, reception_venues_filename)
# data_reception_venues = pd.read_csv(reception_venues_file_path)

reception_venue_field_map = {
    "title": "vendor_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

reception_venue_details_fields = [
    "amenities",
    "ceremony_types",
    "guest_capacity",
    "settings",
    "venue_service_offerings"
]

vectorizer_reception_venue = BaseVectorizer(
    product_table="reception_venue_data",
    csvFilePath=reception_venues_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_receptionvenues,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_receptionvenues,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="venue",
    field_map=reception_venue_field_map,
    details_fields=reception_venue_details_fields,
    extra_fields=["social_urls", "source_url"]
)

FAISS_INDEX_PATH_rehearsaldinners = r'indexes\faiss_index_RehearsalDinners_data.index'
id_INDEX_STORAGE_PATH_rehearsaldinners = r'indexes\id_index_RehearsalDinners_data.npy'
rehearsal_dinners_filename = r'RehearsalDinners_rows.csv'
rehearsal_dinners_file_path = os.path.join(current_dir, rehearsal_dinners_filename)
# data_rehearsal_dinners = pd.read_csv(rehearsal_dinners_file_path)

rehearsal_dinners_field_map = {
    "title": "rehearsal_dinners_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

rehearsal_dinners_details_fields = [
    "settings",
    "venue_service_offerings",
    "wedding_activities"
]

vectorizer_rehearsal_dinners = BaseVectorizer(
    product_table="rehearsal_dinners_data",
    csvFilePath=rehearsal_dinners_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_rehearsaldinners,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_rehearsaldinners,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="rehearsal_dinner",
    field_map=rehearsal_dinners_field_map,
    details_fields=rehearsal_dinners_details_fields,
    extra_fields=["social_urls", "source_url"]
)



FAISS_INDEX_PATH_rentals = r'indexes\faiss_index_Rentals_data.index'
id_INDEX_STORAGE_PATH_rentals = r'indexes\id_index_Rentals_data.npy'
rentals_filename = r'Rentals_rows.csv'
rentals_file_path = os.path.join(current_dir, rentals_filename)
# data_rentals = pd.read_csv(rentals_file_path)

rentals_field_map = {
    "title": "rentals_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

rentals_details_fields = ["rentals_equipment"]

vectorizer_rentals = BaseVectorizer(
    product_table="rentals_data",
    csvFilePath=rentals_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_rentals,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_rentals,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="rentals",
    field_map=rentals_field_map,
    details_fields=rentals_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_transportation = r'indexes\faiss_index_Transportation_data.index'
id_INDEX_STORAGE_PATH_transportation = r'indexes\id_index_Transportation_data.npy'
transportation_filename = r'Transportation_rows.csv'
transportation_file_path = os.path.join(current_dir, transportation_filename)
# data_transportation = pd.read_csv(transportation_file_path)

transportation_field_map = {
    "title": "transportation_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

transportation_details_fields = [
    "destination_weddings",
    "service_staff",
    "transportation",
    "wedding_activities"
]

vectorizer_transportation = BaseVectorizer(
    product_table="transportation_data",
    csvFilePath=transportation_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_transportation,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_transportation,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="transportation",
    field_map=transportation_field_map,
    details_fields=transportation_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_travelspecialists = r'indexes\faiss_index_TravelSpecialists_data.index'
id_INDEX_STORAGE_PATH_travelspecialists = r'indexes\id_index_TravelSpecialists_data.npy'
travel_specialists_filename = r'TravelSpecialists_rows.csv'
travel_specialists_file_path = os.path.join(current_dir, travel_specialists_filename)
# data_travel_specialists = pd.read_csv(travel_specialists_file_path)

travel_specialists_field_map = {
    "title": "travel_specialists_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

travel_specialists_details_fields = [
    "destination_weddings",
    "planning",
    "wedding_activities"
]

vectorizer_travel_specialists = BaseVectorizer(
    product_table="travel_specialists_data",
    csvFilePath=travel_specialists_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_travelspecialists,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_travelspecialists,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="travel_specialists",
    field_map=travel_specialists_field_map,
    details_fields=travel_specialists_details_fields,
    extra_fields=["social_urls", "source_url"]
)

FAISS_INDEX_PATH_weddingbands = r'indexes\faiss_index_WeddingBands_data.index'
id_INDEX_STORAGE_PATH_weddingbands = r'indexes\id_index_WeddingBands_data.npy'
wedding_bands_filename = r'WeddingBands_rows.csv'
wedding_bands_file_path = os.path.join(current_dir, wedding_bands_filename)
# data_wedding_bands = pd.read_csv(wedding_bands_file_path)


wedding_band_field_map = {
    "title": "wedding_band_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

wedding_band_details_fields = [
    "instruments",
    "music_genres",
    "music_services",
    "wedding_activities"
]

vectorizer_wedding_band = BaseVectorizer(
    product_table="wedding_band_data",
    csvFilePath=wedding_bands_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_weddingbands,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_weddingbands,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="wedding_band",
    field_map=wedding_band_field_map,
    details_fields=wedding_band_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_weddingcakes = r'indexes\faiss_index_WeddingCakes_data.index'
id_INDEX_STORAGE_PATH_weddingcakes = r'indexes\id_index_WeddingCakes_data.npy'
wedding_cakes_filename = r'WeddingCakes_rows.csv'
wedding_cakes_file_path = os.path.join(current_dir, wedding_cakes_filename)
# data_wedding_cakes = pd.read_csv(wedding_cakes_file_path)

wedding_cake_field_map = {
    "title": "wedding_cake_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

wedding_cake_details_fields = [
    "cakes_desserts",
    "dietary_options"
]

vectorizer_wedding_cake = BaseVectorizer(
    product_table="wedding_cake_data",
    csvFilePath=wedding_cakes_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_weddingcakes,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_weddingcakes,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="wedding_cake",
    field_map=wedding_cake_field_map,
    details_fields=wedding_cake_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_weddingdance = r'indexes\faiss_index_WeddingDance_data.index'
id_INDEX_STORAGE_PATH_weddingdance = r'indexes\id_index_WeddingDance_data.npy'
wedding_dance_filename = r'WeddingDance_rows.csv'
wedding_dance_file_path = os.path.join(current_dir, wedding_dance_filename)
# data_wedding_dance = pd.read_csv(wedding_dance_file_path)

wedding_dance_field_map = {
    "title": "wedding_dance_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

wedding_dance_details_fields = ["music_genres"]

vectorizer_wedding_dance = BaseVectorizer(
    product_table="wedding_dance_data",
    csvFilePath=wedding_dance_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_weddingdance,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_weddingdance,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="wedding_dance",
    field_map=wedding_dance_field_map,
    details_fields=wedding_dance_details_fields,
    extra_fields=["social_urls", "source_url"]
)



FAISS_INDEX_PATH_weddingofficiants = r'indexes\faiss_index_WeddingOfficiants_data.index'
id_INDEX_STORAGE_PATH_weddingofficiants = r'indexes\id_index_WeddingOfficiants_data.npy'
wedding_officiants_filename = r'WeddingOfficiants_rows.csv'
wedding_officiants_file_path = os.path.join(current_dir, wedding_officiants_filename)
# data_wedding_officiants = pd.read_csv(wedding_officiants_file_path)

wedding_officiants_field_map = {
    "title": "wedding_officiants_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

wedding_officiants_details_fields = [
    "ceremony_types",
    "religious_affiliations",
    "wedding_activities"
]

vectorizer_wedding_officiants = BaseVectorizer(
    product_table="wedding_officiants_data",
    csvFilePath=wedding_officiants_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_weddingofficiants,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_weddingofficiants,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="wedding_officiants",
    field_map=wedding_officiants_field_map,
    details_fields=wedding_officiants_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_wedding_photographers = r'indexes\faiss_index_Wedding_Photographers_data.index'
id_INDEX_STORAGE_PATH_wedding_photographers = r'indexes\id_index_Wedding_Photographers_data.npy'
wedding_photographers_filename = r'Wedding_Photographers_rows.csv'
wedding_photographers_file_path = os.path.join(current_dir, wedding_photographers_filename)
# data_wedding_photographers = pd.read_csv(wedding_photographers_file_path)

wedding_photographer_field_map = {
    "title": "photographer_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

wedding_photographer_details_fields = [
    "destination_wedding",
    "photo_shoot_types",
    "photo_and_video",
    "photo_and_video_styles",
    "wedding_activities"
]

vectorizer_wedding_photographer = BaseVectorizer(
    product_table="wedding_photographer_data",
    csvFilePath=wedding_photographers_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_wedding_photographers,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_wedding_photographers,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="wedding_photographer",
    field_map=wedding_photographer_field_map,
    details_fields=wedding_photographer_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_weddingplanners = r'indexes\faiss_index_WeddingPlanners_data.index'
id_INDEX_STORAGE_PATH_weddingplanners = r'indexes\id_index_WeddingPlanners_data.npy'
wedding_planners_filename = r'WeddingPlanners_rows.csv'
wedding_planners_file_path = os.path.join(current_dir, wedding_planners_filename)
# data_wedding_planners = pd.read_csv(wedding_planners_file_path)

wedding_planner_field_map = {
    "title": "wedding_planner_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

wedding_planner_details_fields = [
    "destination_weddings",
    "planning",
    "wedding_activities"
]

vectorizer_wedding_planner = BaseVectorizer(
    product_table="wedding_planner_data",
    csvFilePath=wedding_planners_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_weddingplanners,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_weddingplanners,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="wedding_planner",
    field_map=wedding_planner_field_map,
    details_fields=wedding_planner_details_fields,
    extra_fields=["social_urls", "source_url"]
)


FAISS_INDEX_PATH_wedding_videographers = r'indexes\faiss_index_Wedding_Videographers_data.index'
id_INDEX_STORAGE_PATH_wedding_videographers = r'indexes\id_index_Wedding_Videographers_data.npy'
wedding_videographers_filename = r'Wedding_Videographers_rows.csv'
wedding_videographers_file_path = os.path.join(current_dir, wedding_videographers_filename)
# data_wedding_videographers = pd.read_csv(wedding_videographers_file_path)

wedding_videographer_field_map = {
    "title": "videographer_name",
    "address": "address",
    "contact_number": "contact_number",
    "website_url": "website_url",
    "vendor_rating": "vendor_rating",
    "reviews_count": "review_count",
    "about": "about",
    "pricing": "pricing",
    "image_urls": "image_urls",
    "reviews": "reviews",
}

wedding_videographer_details_fields = [
    "destination_wedding",
    "photo_and_video",
    "photo_and_video_styles",
    "wedding_activities"
]

vectorizer_wedding_videographer = BaseVectorizer(
    product_table="wedding_videographer_data",
    csvFilePath=wedding_videographers_file_path,
    FAISS_INDEX_PATH=FAISS_INDEX_PATH_wedding_videographers,
    id_INDEX_STORAGE_PATH=id_INDEX_STORAGE_PATH_wedding_videographers,
    embedding_model=OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY),
    entity_name="wedding_videographer",
    field_map=wedding_videographer_field_map,
    details_fields=wedding_videographer_details_fields,
    extra_fields=["social_urls", "source_url"]
)


endTime = time.time()
duration = endTime - startTime
timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
log_entry = f"{timestamp} - Initializing the objects for all vendors took {duration:.2f} seconds\n"



print("Runtime logged successfully.")

# def update_run_time_log(text):
#     # Save log to a text file
#     log_file_path = r"logs\run_time_log.txt"
#     log_dir = os.path.dirname(log_file_path)
#     if log_dir:
#         os.makedirs(log_dir, exist_ok=True)
#     text = f'{time.strftime("%Y-%m-%d %H:%M:%S")} - {text}\n'
#     with open(log_file_path, "a", encoding="utf-8") as f:  # "a" appends instead of overwriting
#         f.write(text)

##update_run_time_log(log_entry)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
responses_dict: Dict[str, Dict[str, Any]] = {}


from datetime import datetime
from pymongo import MongoClient, ASCENDING
from typing import List, Dict, Any, Optional

# -----------------------------------------------------------------------------#
# Mongo helpers
# -----------------------------------------------------------------------------#

def get_db() -> "pymongo.database.Database":
    """
    Returns a cached MongoDB database handle.
    Assumes MONGO_CONNECTION_STRING is in your environment.
    """
    global _DB  # simple module-level cache so we open the connection only once
    try:
        return _DB
    except NameError:
        client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
        _DB = client.get_database()
        return _DB



def save_recommendations_to_chat_recommendations(
    *,
    collection,                 
    user_id: str,
    session_id: str,
    category: str,
    results: List[Dict[str, Any]],
    disliked_titles: Optional[List[str]] = None,
    recommendations_uuid: Optional[str] = None,
) -> None:
    """
    Save recommendations in the following structure:
    {
        "user_id": ...,
        "session_id": ...,
        "created_at": ...,
        "updated_at": ...,
        "recommendations": {
            <vendor_category>: {
                <uuid>: [ ...recommendations... ]
            }
        },
        "disliked_titles": [...]
    }
    """
    from pymongo import ASCENDING
    import uuid as uuid_lib

    collection.create_index([("user_id", ASCENDING), ("session_id", ASCENDING)])

    now = datetime.utcnow()
    if isinstance(results, dict):
        cat_map: Dict[str, List[Dict[str, Any]]] = results
    else:
        cat_map = {category: results}

    # Use the provided recommendations_uuid or generate one
    uuid_val = recommendations_uuid or str(uuid_lib.uuid4())

    # Build the update document for each vendor/category
    set_ops = {}
    for vendor_cat, items in cat_map.items():
        if not isinstance(items, list):
            items = [items]
        if items:
            # recommendations.<vendor_cat>.<uuid> = items
            set_ops[f"recommendations.{vendor_cat}.{uuid_val}"] = items

    update_doc = {
        "$set": {
            "updated_at": now,
        },
        "$setOnInsert": {
            "created_at": now,
        },
    }
    if set_ops:
        update_doc["$set"].update(set_ops)
    if disliked_titles:
        update_doc["$addToSet"] = {"disliked_titles": {"$each": disliked_titles}}

    # Upsert the document
    collection.update_one(
        {"user_id": user_id, "session_id": session_id},
        update_doc,
        upsert=True,
    )
    print("data updated :", {
        "user_id": user_id,
        "session_id": session_id,
        # "recommendations": set_ops,
        "disliked_titles": disliked_titles,
        "uuid": uuid_val,
    })



def save_recommendations(
    *,
    collection,                 # ← pass in db["all recommendations"]
    user_id: str,
    session_id: str,
    # query: str,
    category: str,
    results: List[Dict[str, Any]],
    disliked_titles: Optional[List[str]] = None,
) -> None:
    collection.create_index([("user_id", ASCENDING),
                             ("session_id", ASCENDING)])

    now = datetime.utcnow()
    if isinstance(results, dict):
        cat_map: Dict[str, List[Dict[str, Any]]] = results
    else:
        cat_map = {category: results}

    # ----------------------------------------------------------
    # 2) build the $addToSet operations
    # ----------------------------------------------------------
    add_ops: Dict[str, Any] = {}

    for cat, items in cat_map.items():
        if not isinstance(items, list):
            items = [items]
        if items:
            add_ops[f"recommendations.{cat}"] = {"$each": items}

    if disliked_titles:                       # keep the array unique
        add_ops["disliked_titles"] = {"$each": disliked_titles}

    # ----------------------------------------------------------
    # 3) assemble the update document
    # ----------------------------------------------------------
    update_doc = {
        "$set": {
            "updated_at": now,
        },
        "$setOnInsert": {
            "created_at": now,
        },
    }
    if add_ops:
        update_doc["$addToSet"] = add_ops

    # ----------------------------------------------------------
    # 4) upsert
    # ----------------------------------------------------------
    collection.update_one(
        {"user_id": user_id, "session_id": session_id},
        update_doc,
        upsert=True,
    )
    print("data updated :" , {
        "user_id": user_id,
        "session_id": session_id,
        "category": category,
        # "results": results,
        "disliked_titles": disliked_titles,
        })
    




async def summarise_recommendations(grouped: dict, model: str = "gpt-4o-mini") -> dict[str, list[dict]]:
    system = {
        "role": "system",
        "content": (
            "For EACH wedding vendor below, write a 3-line summary that highlights the location, vibe, pricing, capacity, and standout features. "
            "NEVER repeat the title. NO useless information. Return STRICTLY one JSON object per vendor in this format: "
            "{\"title\": \"<title>\", \"summary\": \"<summary>\"}"
        )
    }

    out: dict[str, list[dict]] = {}
    recommendations = grouped.get("Artifact", {}).get("recommendations", {})

    # Create tasks for concurrent processing
    async def process_vendor(category: str, vendor: dict) -> tuple[str, dict]:
        user = {
            "role": "user",
            "content": f"{json.dumps(vendor, ensure_ascii=False)}"
        }
        try:
            rsp = await async_openai_client.chat.completions.create(
                model=model,
                messages=[system, user],
                temperature=0.6
            )
            content = rsp.choices[0].message.content.strip()
            summary_json = json.loads(content)

            if isinstance(summary_json, dict) and "title" in summary_json and "summary" in summary_json:
                return category, summary_json
            else:
                return category, {
                    "title": vendor.get("title", "Unknown"),
                    "summary": "Model did not return correct format."
                }
        except Exception as e:
            return category, {
                "title": vendor.get("title", "Unknown"),
                "summary": f"Error: {str(e)}"
            }

    # Collect all tasks
    tasks = []
    for category, vendors in recommendations.items():
        print("category in summarise_recommendations:", category)
        out[category] = []
        for vendor in vendors:
            tasks.append(process_vendor(category, vendor))

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Group results by category
    for category, result in results:
        out[category].append(result)

    return out

def get_guest_and_budget_by_user_id(user_id_str , wedding_session_id_str) -> Dict[str, str]:
        """Retrieve guest count and budget for a user based on their user ID and wedding session ID."""
        
        collection = db["weddingSessionOperations"]

        # Convert string to ObjectId
        user_oid = ObjectId(user_id_str)
        wedding_session_oid = ObjectId(wedding_session_id_str)

        # Query to find the document where users.bride == user_id
        document = collection.find_one({"users.bride": user_oid, "_id": wedding_session_oid})

        if not document:
            return {"guest_count": "0", "budget": "0"}

        # Process guest count
        guest_str = document.get("wedding_guests", "0")
        

        # Process budget
        budget_str = document.get("wedding_budget", "$0")
        

        return {
            "guest_count": guest_str,
            "budget": budget_str
        }

def call_perplexity(category: str , session_id:str):
    progress_coll = db["progress"]
    ## get record by session id
    record = progress_coll.find_one({"wedding_session_id" : session_id})
    if not record:
        return {"error": "No record found for the given session ID."}
    # get the progress_flags
    progress_flags = record.get("progress_flags", {})
    if category in progress_flags:
        return True
    return False
async def search(
    query: str,
    user_message: str, 
    category: str, 
    location: str,
    user_id: str,
    wedding_session_id: str,
    disliked_titles: Optional[list] = None,
    *,
    attempt: int = 0, 
    max_attempts: int = 2
) -> Dict[str, Any]:
    """Clean Perplexity-only search function"""
    
    start_search_function = time.time()
    
    print(f"Starting search for category: {category}, location: {location}")
    
    if disliked_titles is None:
        disliked_titles = []
    
    collection = db["all_recommendations"]
    chat_recommendation_collection = db["chat_recommendations"]
    
    # Call Perplexity API directly with debugging
    print("Calling Perplexity API...")
    
    # Debug API key
    api_key = os.getenv("PERPLEXITY_API_KEY")
    print(f"API Key exists: {bool(api_key)}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    
    perplexity_start_time = time.time()
    
    try:
        perplexity_results = await get_vendor_recommendations_with_details_perplexity(
            user_message, category, location, disliked_titles
        )
        
        perplexity_end_time = time.time()
        print(f"Perplexity API took {perplexity_end_time - perplexity_start_time:.2f} seconds")
        print(f"Perplexity returned {len(perplexity_results)} results")
        
    except Exception as e:
        print(f"Perplexity API failed: {e}")
        perplexity_results = []
    
    # Handle empty results with retry logic
    if not perplexity_results and attempt < max_attempts:
        print(f"No results from Perplexity API, retrying... (attempt {attempt + 1}/{max_attempts})")
        
        # Retry with modified search terms (using original format)
        retry_queries = [
            f"{category} for weddings near {location}",  # Broader location
            f"wedding {category} {location}",            # Different phrasing
            f"{category} vendors {location}",            # Generic approach
        ]
        
        for retry_query in retry_queries:
            print(f"Retrying with query: {retry_query}")
            try:
                retry_results = await get_vendor_recommendations_with_details_perplexity(
                    retry_query, category, location, disliked_titles
                )
                if retry_results:
                    print(f"Retry successful! Found {len(retry_results)} results")
                    perplexity_results = retry_results
                    break
            except Exception as e:
                print(f"Retry failed: {e}")
                continue
        
        # If still no results after retries, try recursive call with incremented attempt
        if not perplexity_results:
            print("All retries failed, making recursive attempt...")
            return await search(
                query, user_message, category, location, 
                user_id, wedding_session_id, disliked_titles,
                attempt=attempt + 1, max_attempts=max_attempts
            )
    
    # Final fallback after all attempts
    if not perplexity_results:
        print(f"No results after {max_attempts} attempts")
        return {
            "Artifact": {
                "error": f"No results found after {max_attempts} attempts",
                "category": category,
                "query": query,
                "recommendations": {}
            }
        }
    
    # Limit to top 8 results (no duplicates since we only call once)
    final_results = perplexity_results[:8]
    print(f"Final results count: {len(final_results)}")
    
    # Generate UUID for tracking
    import uuid
    recommendations_uuid = str(uuid.uuid4())
    
    # Format results properly
    formatted_results = {
        "Artifact": {
            "recommendations": {category: final_results},
            "recommendations_uuid": recommendations_uuid
        }
    }
    
    # Save to MongoDB
    start_save_time = time.time()
    
    # Save to JSON file in json format for debugging
    # with open(r"H:\PGAGI\SayYesAI_Backend\logs\teminal_output.json", "w", encoding="utf-8") as f:
    #     json.dump(formatted_results , f, indent=4, ensure_ascii=False)


    try:
        save_recommendations(
            collection=collection,
            user_id=user_id,
            session_id=wedding_session_id,
            category=category,
            results=formatted_results["Artifact"]["recommendations"], 
            disliked_titles=disliked_titles
        )

        save_recommendations_to_chat_recommendations(
            collection=chat_recommendation_collection,
            user_id=user_id,
            session_id=wedding_session_id,
            category=category,
            results=formatted_results["Artifact"]["recommendations"], 
            disliked_titles=disliked_titles,
            recommendations_uuid=recommendations_uuid
        )
        
        end_save_time = time.time()
        print(f"MongoDB save took {end_save_time - start_save_time:.2f} seconds")
        
    except Exception as e:
        print(f"MongoDB save failed: {e}")
        # Continue anyway, don't fail the entire request
    
    end_search_function = time.time()
    print(f"Total search function took {end_search_function - start_search_function:.2f} seconds")
    
    return formatted_results




def sort_recommendations_on_addresses(
    user_query: str,
    recommendations: List[Dict],
    user_location: str,
    top_k: int = 3,
    model: str = "gpt-4o-mini"
) -> List[int]:
    user_vibe = user_query
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # 1. Extract top 3 indexed records
    indexed_recs = [
        {"index": i,"title": rec.get("title", ""), "address": rec.get("address", ""),
         "pricing": rec.get("pricing", ""),
         "vendor_rating": rec.get("vendor_rating", ""),
         "details": rec.get("details",""),
         "about": rec.get("about", "")}
        for i, rec in enumerate(recommendations)
    ]
    print(f"Top 3 candidate records: {indexed_recs}")

    # 2. System Instruction
    system_prompt = (
        """
        You are a vendor filtering and ranking assistant.
        Your task is to analyze vendor records against user requirements and return vendors that can accommodate the user's needs.
        
        **Important**: For style/theme requests (like "Indian style"), consider that most venues can accommodate different cultural wedding styles through decoration, catering, and setup. Don't exclude venues that don't explicitly mention the style.
        
        **Filtering Criteria:**
        1. **Location Match**: The vendor's address must be in or reasonably near the user's desired location
        2. **Service Capability/ Theme Match**: The vendor should be able to accommodate/align the user's requirements (venue capacity, style flexibility, theme etc.)
        
        **Instructions:**
        - Be more inclusive rather than exclusive for style-based requests
        - Prioritize venues that explicitly support the requested style, but don't exclude others
        - Consider venue capacity, settings, and amenities that would work for the requested style
        - Focus on practical suitability rather than exact style matches
        
        **Response Format:**
        Respond ONLY with a JSON array of indices, sorted from best match to least best match:
        [index1, index2, index3, index4, index5]
        Do not provide explanations, reasoning, or any text outside the JSON array.
        
        """
    )

    # 3. Build user prompt using the specified format
    prompt_lines = [
        f"User Query: {user_query}",
        f"User Desired Location: {user_location}",
        f"User Desired Vibe: {user_vibe}",
        ""
    ]

    for item in indexed_recs:
        prompt_lines.append(
            f"{item['index']}: "
            f"title={item['title']} | "
            f"addr={item['address']} | "
            f"details={item['details']} | "
            f"rating={item['vendor_rating'] or 'NA'} | "
            f"pricing={item['pricing'] or 'NA'} | "
            f"about={item['about'] or 'NA'}"
        )

    full_prompt = "\n".join(prompt_lines)

    # print("Full prompt for OpenAI API:\n", full_prompt)
    # 4. Call OpenAI API
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return [], []

    # 5. Parse output
    content = resp.choices[0].message.content.strip()
    # print("Model response:", content[:1000])  # Print first 1000 characters for debugging

    try:
        selected_indices = json.loads(content)
    except json.JSONDecodeError:
        print("JSON decode failed. Falling back to manual parsing.")

        import re

        # Remove possible markdown formatting
        cleaned = content.replace("```json", "").replace("```", "").strip()

        # Try to extract a list from within square brackets
        match = re.search(r"\[(.*?)\]", cleaned)

        if match:
            # Parse numbers inside the brackets
            selected_indices = [int(i.strip()) for i in match.group(1).split(",") if i.strip().isdigit()]
        else:
            # Fallback: extract all standalone digits from the text
            selected_indices = [int(tok) for tok in cleaned.split() if tok.isdigit()]



    # 6. Fetch top records
    selected_records = [recommendations[i] for i in selected_indices if i < len(recommendations)]

    # print("Selected indices:", selected_indices)
    # print("Selected records:", selected_records)

    return selected_indices, selected_records



def remove_duplicates_by_title(data: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    result = {}
    ##update_run_time_log("remove_duplicates_by_title function called")
    print(f"number of records bedore deduplication : {len(list(data.values()))}")
    for category, records in data.items():
        seen_titles = set()
        unique_records = []

        for record in records:
            title = record.get("title")
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_records.append(record)

        result[category] = unique_records ## considering only 3 unique records
    # ##update_run_time_log(f"records : {result}")
    return result


# ----------------------------------------------------------
# Adding google places api results
# ----------------------------------------------------------

import requests

GOOGLE_PLACES_API_KEY = "AIzaSyB6n5d427zt4w9WKvknyBZOPQGIrx8gAqY" 

def _get_base_url(url: str) -> str:
    """Extracts base URL (strip query params)."""
    if url:
        return url.split("?")[0]
    return None

def fetch_top_places_from_places_api(query: str, category: str, top_k: int = 3):
    search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    photo_base_url = "https://maps.googleapis.com/maps/api/place/photo"

    # Step 1: Search Places
    response = requests.get(search_url, params={"query": query, "key": GOOGLE_PLACES_API_KEY})
    places = response.json().get("results", [])[:top_k]

    google_api_results = []

    for place in places:
        place_id = place.get("place_id")
        detail_response = requests.get(details_url, params={"place_id": place_id, "key": GOOGLE_PLACES_API_KEY})
        detail = detail_response.json().get("result", {})

        title = place.get("name")
        address = place.get("formatted_address")
        contact_number = detail.get("formatted_phone_number")
        website_url = _get_base_url(detail.get("website"))
        vendor_rating = place.get("rating")
        reviews_count = place.get("user_ratings_total")
        about = detail.get("editorial_summary", {}).get("overview")
        pricing = detail.get("price_level")

        # Opening hours and business details
        details_block = {
            "types": detail.get("types"),
            "business_status": detail.get("business_status"),
            "opening_hours": detail.get("opening_hours", {}).get("weekday_text"),
            "maps_url": detail.get("url")
        }

        # Up to 3 image URLs
        photos = detail.get("photos", [])[:3]
        image_urls = [
            f"{photo_base_url}?maxwidth=1600&photoreference={photo['photo_reference']}&key={GOOGLE_PLACES_API_KEY}"
            for photo in photos
        ]

        # Up to 5 reviews (API limit)
        reviews = [
            {
                "author": r.get("author_name"),
                "rating": r.get("rating"),
                "text": r.get("text"),
                "time": r.get("relative_time_description")
            }
            for r in detail.get("reviews", [])[:5]
        ]

        result_obj =  { category :
                       {
                    "title": title,
                    "address": address,
                    "contact_number": contact_number,
                    "website_url": website_url,
                    "vendor_rating": vendor_rating,
                    "reviews_count": reviews_count,
                    "about": about,
                    "pricing": pricing,
                    "details": details_block,
                    "image_urls": image_urls,
                    "reviews": reviews
                }
        }

        # You can replace `title` with an entity name if needed (e.g., 'venue')
        google_api_results.append(result_obj)
    # update_run_time_log(f"google api results fetched for query : {query} , total results : {google_api_results}")
    return google_api_results






def bucket_by_category(raw: list[dict]) -> dict[str, list[dict]]:
    """
    Converts
        [{'Venues': {...}}, {'Venues': {...}}, {'Caterers': {...}}]
    into
        {'venues': [{...}, {...}], 'caterers': [{...}]}
    """
    
    out: dict[str, list] = {}
    for item in raw:
        if not item:          # safety
            continue
        cat, data = next(iter(item.items()))        # e.g. 'Venues', {...}
        key = cat.lower()                           # → 'venues'
        out.setdefault(key, []).append(data)        # stash
    return out




if __name__ == "__main__":
    import asyncio
    import json
    
    async def test_perplexity():
        query = "Bollywood themed"
        category = "djs"
        location = "ausin, texas"
        disliked_titles = ["DJ Awesome", "DJ Cool Beats"]
        results = await get_vendor_recommendations_with_details_perplexity(query, category, location,disliked_titles)
        
        if results:
            print(f"\n✅ Found {len(results)} vendors:\n")
            print(json.dumps(results, indent=2))
        else:
            print("❌ No results returned")
    
    # asyncio.run(test_perplexity())

    query = "beach side wedding venues in manhattan"
    category = "venues"
    location = "Manhattan"
    disliked_titles = []

    user_id = "6899da790501a5cf61755967"
    wedding_session_id = "6899da790501a5cf61755967"
    start_search_time = datetime.now()
    asyncio.run(search(query, query, category, location, user_id, wedding_session_id, disliked_titles))
    elapsed_seconds = (datetime.now() - start_search_time).total_seconds()
    print(f"time taken for the search: {elapsed_seconds:.2f} seconds")
