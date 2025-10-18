from google.oauth2 import service_account
from google.cloud import storage
import asyncio
import datetime as dt
import json
import logging
import math
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import (Any, Dict, List, Optional, Set, Tuple, Union)
from uuid import uuid4

import aiohttp
import boto3
import firebase_admin
import openai
from bson import Decimal128, ObjectId
from bson import errors as bson_errors
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import (Body, FastAPI, File, Header, HTTPException, Path,
                     Query, Request, UploadFile, WebSocketDisconnect
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials
from models import (EmailLoginRequest, FirebaseLoginRequest,
                    Token, UserInfo, WeddingInfo)
from parentAgent import parentAgent
from pydantic import BaseModel, Field
from pymongo import MongoClient
from rag_sayyes import (  # Import from your file
    OptimizedPerplexityVendorSearch, summarise_recommendations
)
# importing notification helper functions
from services import (  # Core database utilities; Segments & chats; Notifications; Agent-related; Progress tracking
    convert_object_ids, create_progress_document, get_collection, get_db,
    get_onboarding_status, get_segment_chat, get_segment_chats,
    get_segments_for_user, notify_session_about_new_member,
    send_agent_response, send_bride_reminder_notification,
    send_feed_interaction_notification, send_image_upload_notification,
    send_member_joined_notification, send_notification,
    send_reaction_notification_webhook, send_webhook_to_session_members)
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket

# def update_run_time_log(text):
#     # Save log to a text file
#     log_file_path = r"logs\run_time_log.txt"
#     log_dir = os.path.dirname(log_file_path)
#     if log_dir:
#         os.makedirs(log_dir, exist_ok=True)
#     text = f'{time.strftime("%Y-%m-%d %H:%M:%S")} - {text}\n'
#     with open(log_file_path, "a", encoding="utf-8") as f:  # "a" appends instead of overwriting
#         f.write(text)

load_dotenv()

app = FastAPI()

websocket_last_seen = {}  # key = session_key, value = datetime

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI API key and MongoDB connection
openai.api_key = os.getenv("OPENAI_API_KEY")
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_CONNECTION_STRING)
db = client.get_database()
mood_board_collection = db.mood_board
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "sayyes-image-storage"
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=AWS_REGION,
)
IST = timezone(timedelta(hours=5, minutes=30))

user_last_activity = {}  # Track last activity time per user
user_disconnect_time = {}  # Track when user disconnected

now_ist = datetime.now(IST)

BEARER_TOKEN = os.getenv("BEARER_TOKEN")
SERVICE_CRED = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Task(BaseModel):
    taskName: str
    assignee: Optional[str] = None
    taskDescription: Optional[str] = None
    taskStatus: str = Field(
        default="pending",
        pattern="^(pending|doing|done)$")
    assignedDate: str
    dueDate: str
    id: str = "string"  # default value

    # Convert Task instance to dictionary
    def to_dict(self):
        return self.dict()


class FCMTokenUpdate(BaseModel):
    fcm_token: str = Field(..., description="Firebase Cloud Messaging token")


class Sentence(BaseModel):
    text: str


class TaskCreateInput(BaseModel):
    task: Task


openai.api_key = os.getenv("OPENAI_API_KEY")
user_collection = db["user_operations"]
seen_users = set()

cred = credentials.Certificate(SERVICE_CRED)
firebase_admin.initialize_app(cred)


def generate_bearer_token() -> str:
    return secrets.token_hex(32)


user_websockets: Dict[str, WebSocket] = {}

LEFT_CHAT_DELAY = 30  # Delay in minutes before considering a user has left the chat
DELAY_AFTER_RECOMMENDATIONS = (
    60  # it is considered in minutes, for 3 days use 60* 24 * 3
)

# Track user activity and idle/left status
user_activity_log: Dict[str, list] = {}
user_last_seen: Dict[str, datetime] = {}
task_registry: Dict[str, Dict[str, Any]] = {}

idle_notified: Dict[str, bool] = {}
left_triggered: Dict[str, bool] = {}
idle_intervals_logged: Dict[str, int] = {}
has_connected_before: Set[Tuple[str, str]] = set()
# -----------------------------------
# Utility functions for activity logging and tracking

# -----------------------------------

# import gridfs
# from fastapi import UploadFile, File
# from fastapi.responses import StreamingResponse
#
#  fs = gridfs.GridFS(db)

# @app.post("/upload-image/")
# async def upload_image(file: UploadFile = File(...),
#     bearer_token: str = Header(..., description="Your existing bearer token")
# ):

#     user = authenticate_user(bearer_token)
#     user_id = user["_id"]

#     # Read the image data as bytes
#     image_data = await file.read()

#     # Save the image to GridFS and get the file_id
#     file_id = fs.put(image_data, filename=file.filename)

#     # Return the file_id so it can be used to retrieve the file
#     return {"file_id": str(file_id)}

# @app.get("/get-image/{file_id}")
# async def get_image(file_id: str):
#     # Convert the string file_id to ObjectId
#     try:
#         grid_out = fs.get(ObjectId(file_id))  # Retrieve the image using the file_id
#     except gridfs.errors.NoFile:
# return {"error": "File not found"}  # Return error if the file doesn't
# exist

#     # Get the media type from the filename (for example, ".jpg", ".png", etc.)
#     file_extension = grid_out.filename.split('.')[-1].lower()
#     if file_extension == 'jpg' or file_extension == 'jpeg':
#         media_type = "image/jpeg"
#     elif file_extension == 'png':
#         media_type = "image/png"
#     elif file_extension == 'gif':
#         media_type = "image/gif"
#     else:
#         media_type = "application/octet-stream"  # Default for unknown types

#     # Return the image as a streaming response
#     return StreamingResponse(grid_out, media_type=media_type)


def authenticate_user(bearer_token: str):
    """Authenticate the user using the Bearer token."""
    # Check if the bearer_token is valid in the userOperations collection
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return user


def log_activity(user_id: str, activity_type: str, details: str):
    """Log user activities and update last seen timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    activity = {
        "timestamp": datetime.utcnow().isoformat(),
        "activity_type": activity_type,
        "details": details,
    }
    user_activity_log.setdefault(user_id, []).append(activity)
    print(f"Activity logged for user {user_id}: {activity_type} - {details}")
    user_last_seen[user_id] = datetime.now()


def get_user_activity(user_id: str):
    """Retrieve activity logs for a user."""
    return user_activity_log.get(user_id, [])


def user_idle(user_id: str):
    """Calculate how long a user has been idle."""
    idle_time = datetime.now() - user_last_seen.get(user_id, datetime.now())
    print(f"User {user_id} idle for {idle_time}.")
    return idle_time


async def notify_parent_of_task_status(
    user_id: str, task_id: str, task_type: str, completed: bool
):
    status_msg = "completed" if completed else "not completed (timeout)"
    print(
        f"Notifying parent agent: Task {task_id} for user {user_id} is {status_msg}")

    db["taskStatusUpdates"].insert_one(
        {
            "user_id": user_id,
            "task_id": task_id,
            "task_type": task_type,
            "status": status_msg,
            "timestamp": datetime.now(),
        }
    )

    # Send live notification if websocket is connected
    websocket = user_websockets.get(user_id)
    if websocket:
        try:
            await send_notification(websocket, f"Task {task_id} {status_msg}.")
        except Exception as e:
            print(f"Failed to send live notification to user {user_id}: {e}")


def notify_parent_agent(user_id: str, callback):
    activity_log = get_user_activity(user_id)
    print(
        f"Notifying parent agent about user {user_id}'s activities: {activity_log}")
    callback(user_id, activity_log)


# async def send_initial_recommendations(
#     websocket,
#     user_id: str,
#     wedding_session_id: str,
#     responses_dict: dict,
#     prefetched_recommendations: dict | None = None,
# ):
#     """
#     Send initial recommendations when the WebSocket connects.

#     Only sends recommendations on first connection (when chat history is empty).
#     Shows 4 random recommendations from each category out of 10 available.
#     """
# async def process_and_send_result(result, websocket, responses_dict, wedding_session_id, user_id):
#     """Process and send recommendation results to the user (unchanged contract)."""
#     try:
#         if not result.get("success") or not result.get("results"):
#             print(f"No results to process for {result.get('category')}")
#             return

#         category = result["category"]
#         raw_results = result["results"]

#         # Extract the actual recommendations array from the nested structure
#         if isinstance(raw_results, dict) and "Artifact" in raw_results:
#             artifact_content = raw_results["Artifact"].copy()
#             recommendations_uuid = artifact_content.get("recommendations_uuid")

#             formatted_results = {
#                 "ResponseComplete": {
#                     "type": "recommendation_agent",
#                     "content": json.dumps({"Artifact": artifact_content}, ensure_ascii=False),
#                     "segment": category.title()
#                 }
#             }
#             print(f"Formatted results for {category} with Artifact and segment: {category.title()}")
#         else:
#             print(f"Unexpected result structure for {category}")
#             return

#         # Send to websocket
#         if websocket.client_state.name != 'DISCONNECTED':
#             await websocket.send_text(json.dumps(formatted_results))
#         else:
#             return

#         # Add to conversation history (store the formatted response)
#         responses_dict["conversation_history"].append({
#             "role": "assistant",
#             "content": json.dumps(formatted_results),
#             "timestamp": datetime.utcnow().isoformat()
#         })

#         print(f"Successfully sent {category} recommendations with UUID: {recommendations_uuid}")

#     except Exception as e:
#         print(f"Error processing result for {result.get('category', 'unknown')}: {e}")

# def _is_first_connection(responses_dict: dict) -> bool:
#     """Check if this is the first connection by examining chat history."""
#     conversation_history = responses_dict.get("conversation_history", [])

#     # If no conversation history, it's definitely first connection
#     if not conversation_history:
#         return True

#     # Check if there are any assistant messages with recommendations
#     for message in conversation_history:
#         if message.get("role") == "assistant":
#             content = message.get("content", "")
#             # Check if content contains recommendation data
#             if isinstance(content, str):
#                 try:
#                     parsed_content = json.loads(content)
#                     if (parsed_content.get("ResponseComplete", {}).get("type") == "recommendation_agent" or
#                         "recommendation" in content.lower()):
#                         return False
#                 except json.JSONDecodeError:
#                     if "recommendation" in content.lower():
#                         return False

#     return True

# async def _send_cached_random_picks_from_files():
#     """
#     Load each category from separate saved files and send 4 random picks per category.
#     Store summaries in conversation history for chatbot context.
#     """
#     try:
#         categories = ["venues", "bridal_salons", "photographers"]
#         all_summaries = {}

#         # Get database connection
#         db = get_db()

#         for category in categories:
#             try:
#                 # Load category-specific file
#                 filename = f"{category}_recommendations.json"
#                 file_path = os.path.join("recommendations", filename)

#                 print(f"Loading {category} recommendations from {file_path}")

#                 # Load the JSON file with UTF-8 encoding
#                 with open(file_path, 'r', encoding='utf-8') as f:
#                     category_data = json.load(f)

#                 # Extract recommendations from the loaded file
#                 content = json.loads(category_data["ResponseComplete"]["content"])
#                 items = content["Artifact"]["recommendations"][category]

#                 # Pick 4 random items
#                 k = min(4, len(items))
#                 if k == 0:
#                     print(f"No items for category '{category}', skipping.")
#                     continue

#                 picks = random.sample(items, k=k)

#                 # Create category-specific artifact
#                 cat_artifact = {
#                     "recommendations": {category: picks},
#                     "recommendations_uuid": str(uuid.uuid4())
#                 }

#                 result = {
#                     "category": category,
#                     "results": {"Artifact": cat_artifact},
#                     "success": True
#                 }

#                 # Send the recommendations
#                 await process_and_send_result(result, websocket, responses_dict, wedding_session_id, user_id)
#                 print(f"Successfully sent {category} recommendations")

#                 # Save to database so they can be found later for saving to mood board
#                 try:
#                     user_id_for_save = str(user_id)

#                     save_recommendations(
#                         collection=db["all_recommendations"],
#                         user_id=str(user_id),  # Ensure user_id is string
#                         session_id=wedding_session_id,
#                         category=category,
#                         results=cat_artifact["recommendations"],  # Just the recommendations part
#                         disliked_titles=[]
#                     )
#                     print(f"Saved {category} initial recommendations to database")

#                     # DEBUG: Verify the data was actually saved
#                     saved_doc = db["all_recommendations"].find_one({
#                         "user_id": str(user_id),
#                         "session_id": wedding_session_id
#                     })
#                     if saved_doc:
#                         print(f"DEBUG: Found saved document with categories: {list(saved_doc.get('recommendations', {}).keys())}")
#                         print(f"DEBUG: User ID in saved doc: {saved_doc.get('user_id')}")
#                         print(f"DEBUG: Session ID in saved doc: {saved_doc.get('session_id')}")
#                     else:
#                         print(f"DEBUG: No saved document found for user_id={str(user_id)}, session_id={wedding_session_id}")

#                 except Exception as save_error:
#                     print(f"Error saving {category} to database: {save_error}")

#                 # Generate summaries for chatbot context
#                 try:
#                     print(f"Generating summaries for {category}")
#                     summaries = await summarise_recommendations({"Artifact": cat_artifact})
#                     if summaries:
#                         all_summaries[category] = summaries
#                         print(f"Generated summaries for {category}: {summaries}")
#                 except Exception as summary_error:
#                     print(f"Error generating summaries for {category}: {summary_error}")

#             except FileNotFoundError:
#                 print(f"File not found: {filename}")
#                 continue
#             except UnicodeDecodeError as e:
#                 print(f"Unicode decode error for {filename}: {e}")
#                 continue
#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error for {filename}: {e}")
#                 continue
#             except Exception as category_error:
#                 print(f"Error loading {category} recommendations: {category_error}")
#                 continue

#         # After sending all recommendations, add context to conversation history
#         if all_summaries:
#             print("Adding initial recommendation summaries to conversation history")

#             # Add summaries to conversation history for chatbot context
#             title_summary = {
#                 "role": "system",
#                 "content": json.dumps(all_summaries, ensure_ascii=False),
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#             responses_dict["conversation_history"].append(title_summary)

#             # Add system instruction for acknowledgment
#             system_instruction = {
#                 "role": "system",
#                 "content": {
#                     "type": "system",
#                     "chat": "Initial recommendations have been sent to the user! You just sent recommendations for multiple categories. Based on the location/s of the vendors acknowledge that you have given recommendations in those particular locations (ensure to mention all the locations). If the location of the vendor doesn't match any requested location, let them know what locations you were able to find. Please ask for validation or feedback on the recommendations."
#                 },
#                 "timestamp": datetime.utcnow().isoformat()
#             }
#             responses_dict["conversation_history"].append(system_instruction)

#             print("Added summaries and system instruction to conversation history")

#             # Call parentAgent to generate acknowledgment response
#             print("Calling parentAgent for initial recommendation acknowledgment")
#             try:
# acknowledgment_resp = await parentAgent(websocket, responses_dict,
# user_id, wedding_session_id, watcher_event=True)

#                 if acknowledgment_resp:
#                     # Add to conversation history
#                     responses_dict["conversation_history"].append({
#                         "role": "assistant",
#                         "content": json.dumps(acknowledgment_resp, ensure_ascii=False) if acknowledgment_resp is not None else "",
#                         "timestamp": datetime.utcnow().isoformat()
#                     })

#                     # Send acknowledgment response
#                     print("Sending acknowledgment response for initial recommendations")
#                     await send_agent_response(websocket, acknowledgment_resp)

#             except Exception as ack_error:
#                 print(f"Error generating acknowledgment response: {ack_error}")

#     except Exception as e:
#         print(f"Error in _send_cached_random_picks_from_files: {e}")

# try:
#     # Check if this is the first connection
#     if not _is_first_connection(responses_dict):
#         print("Not first connection - skipping initial recommendations")
#         return

#     # Fetch wedding details (kept for logging/fallback)
#     wedding_collection = db["weddingSessionOperations"]
#     wedding_details = wedding_collection.find_one({"_id": ObjectId(wedding_session_id)})

#     if not wedding_details:
#         print("No wedding details found for initial recommendations")

#     # List of 10 different welcome messages
#     welcome_messages = [
#         "Welcome! I've prepared some amazing recommendations based on your wedding details!",
#         "Hello! I've curated some fantastic options tailored to your special day!",
#         "Hi there! I've gathered some wonderful suggestions perfect for your wedding!",
#         "Welcome back! I've handpicked some incredible recommendations just for you!",
#         "Great to see you! I've found some beautiful options for your celebration!",
#         "Hello! I've selected some perfect choices based on your wedding preferences!",
#         "Welcome! I've compiled some stunning recommendations for your big day!",
#         "Hi! I've discovered some amazing vendors that match your wedding vision!",
#         "Welcome! I've organized some fantastic options to make your day special!",
#         "Hello there! I've prepared some incredible suggestions for your dream wedding!"
#     ]

#     # Randomly select a welcome message
#     welcome_message = random.choice(welcome_messages)

#     if websocket.client_state.name != 'DISCONNECTED':
# await websocket.send_text(json.dumps({"ResponseStatus": "Bot response
# started"}))

#         # Stream the welcome message word by word
#         words = welcome_message.split()
#         chunk_buffer, chunk_counter, chunk_threshold = [], 0, 3

#         for i in range(0, len(words), 1):
#             chunk = " ".join(words[i:i+1]) + " "
#             chunk_buffer.append(chunk)
#             chunk_counter += 1
#             if chunk_counter >= chunk_threshold:
#                 combined_chunk = ''.join(chunk_buffer)
#                 await websocket.send_text(json.dumps({"botResponseStream": combined_chunk}))
#                 chunk_buffer, chunk_counter = [], 0

#         # Send any remaining chunks
#         if chunk_buffer:
#             combined_chunk = ''.join(chunk_buffer)
# await websocket.send_text(json.dumps({"botResponseStream":
# combined_chunk}))

# await websocket.send_text(json.dumps({"ResponseStatus": "Bot response
# ended"}))

#         # Add to conversation history
#         responses_dict.setdefault("conversation_history", []).append({
#             "role": "assistant",
#             "content": welcome_message,
#             "timestamp": datetime.utcnow().isoformat()
#         })

#         # Show loading spinner
#         await websocket.send_text(json.dumps({"type": "startLoading"}))
#         # Add 2 second delay before showing loading and sending recommendations
#         await asyncio.sleep(2)
#     else:
#         return

#     # Load recommendations from saved files
#     print("Loading recommendations from saved files")
#     await _send_cached_random_picks_from_files()

#     # # -------- Send final message (randomly selected) --------
#     # if websocket.client_state.name == 'DISCONNECTED':
#     #     return

#     # # List of 10 similar final messages
#     # final_messages = [
#     #     "Here are your personalized recommendations! Let me know which ones catch your eye!",
#     #     "I've curated these special recommendations just for you! Tell me what you think!",
#     #     "Check out these handpicked suggestions for your wedding! Which ones speak to you?",
#     #     "Here are some amazing options I found for your big day! What catches your attention?",
#     #     "I've gathered these wonderful recommendations tailored to your wedding! Share your thoughts!",
#     #     "Take a look at these carefully selected options for your celebration! Any favorites?",
#     #     "Here are some fantastic choices I've picked out for your wedding! What do you think?",
#     #     "I've found these perfect recommendations just for your special day! Which ones do you love?",
#     #     "Check out these beautiful options I've selected for your wedding! Tell me your favorites!",
#     #     "Here are some incredible recommendations designed for your celebration! What appeals to you?"
#     # ]

#     # # Randomly select one message
#     # final_message = random.choice(final_messages)

#     # final_response = {
#     #     "ResponseComplete": {
#     #         "content": final_message,
#     #         "segment": None
#     #     }
#     # }

#     # if websocket.client_state.name != 'DISCONNECTED':
#     #     await websocket.send_text(json.dumps(final_response))
#     # else:
#     #     return

#     # responses_dict["conversation_history"].append({
#     #     "role": "assistant",
#     #     "content": final_message,
#     #     "timestamp": datetime.utcnow().isoformat()
#     # })

# except Exception as e:
#     print(f"Error in send_initial_recommendations: {e}")


async def dynamic_watcher_function():
    while True:
        # print("hiii")
        now = datetime.now()
        to_remove = []

        for session_key, last_seen in websocket_last_seen.items():
            print(f"Checking session {session_key}. last_seen={last_seen}")

            if not last_seen:
                print(f"[DISCONNECTED] {session_key}: never seen activity.")
                continue

            disconnected = session_key not in user_websockets

            if disconnected and datetime.now() - last_seen > timedelta(seconds=30):
                print(
                    f"[DISCONNECTED] WebSocket for session {session_key} has been disconnected for over 30 seconds."
                )

            if disconnected:
                notification1 = datetime.now() - last_seen > timedelta(minutes=30)
                if notification1:
                    title = "Don't forget to check in!"
                    remainder_text = "It's been 30 minutes since you last spoke to me."
                    await send_bride_reminder_notification(
                        session_key, title, remainder_text
                    )

                notification2 = datetime.now() - last_seen > timedelta(hours=8)
                if notification2:
                    title = "Don't forget to check in!"
                    remainder_text = "It's been 8 hrs since you last spoke to me."
                    await send_bride_reminder_notification(
                        session_key, title, remainder_text
                    )

        for task_id, task in list(task_registry.items()):
            # Only act on active tasks once they time out
            if task["status"] == "active":
                elapsed = now - task["start_time"]
                if (
                    elapsed > timedelta(minutes=task["timeout_minutes"])
                    and not task["notified_parent"]
                ):

                    print(f"User {task['user_id']}")
                    params = task["params"]
                    print(f"Params : {params}")
                    uid = task["params"]["user_id"]
                    # Use get() with default value for wedding_session_id to
                    # handle cases where it's not present
                    wedding_session_id = task["params"]["wedding_session_id"]
                    session_key = f"{uid}_{wedding_session_id}"
                    websocket = user_websockets.get(session_key)
                    print("websocket------------", websocket)

                    # 1) Idle event → let the LLM choose how to follow up
                    if task["task_type"] == "handle_idle_event":
                        # Signal the idle event to Bella
                        responses_dict[session_key]["conversation_history"].append(
                            {
                                "role": "system",
                                # Save content as a string
                                "content": f"User has been idle for {task['params']['idle_duration']}",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                        print(
                            f"Handling idle event for user {
                                task['user_id']} with idle duration {
                                task['params']['idle_duration']}"
                        )
                        # Call back into the LLM
                        resp = await parentAgent(
                            websocket,
                            responses_dict[session_key],
                            user_id=None,
                            wedding_session_id=None,
                            watcher_event=True,
                        )
                        # Send Bella's reply
                        if websocket:
                            await send_agent_response(websocket, resp)
                        stop_task(task_id)

                    # 2) Left-chat event → LLM crafts the message
                    elif task["task_type"] == "user_left_chat":
                        responses_dict[session_key]["conversation_history"].append(
                            {
                                "role": "system",
                                "content": {
                                    "type": "system: ",
                                    "chat": "User has left the chat.",
                                },
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                        resp = await parentAgent(
                            websocket,
                            responses_dict[session_key],
                            user_id=None,
                            wedding_session_id=None,
                            watcher_event=True,
                        )
                        print("resp left------------", resp)
                        if websocket:
                            await send_agent_response(websocket, resp)
                        stop_task(task_id)

                    elif task["task_type"] == "Remainder_after_3days":
                        print(
                            "entered remainder||||||||||||||||||||||||||||||||||||||||||||||"
                        )
                        responses_dict[session_key]["conversation_history"].append(
                            {
                                "role": "system",
                                "content": {
                                    "type": "system: ",
                                    "chat": "Remainder_after_three_days",
                                },
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                        # resp = await parentAgent(websocket, responses_dict[uid],user_id=None,wedding_session_id=None, watcher_event=True)
                        # <here send mail>
                        print("uid------------", uid)
                        user_doc = db["userOperations"].find_one(
                            {"_id": ObjectId(uid)},  # <-- search by _id
                            {"_id": 0, "email": 1},  # projection: e-mail only
                        )
                        print("user_doc------------", user_doc)
                        # assignee_email = user_doc.get("email") if user_doc else None
                        # print("assignee_email------------",assignee_email)
                        # if assignee_email:
                        #     asyncio.create_task(
                        #         send_email_async(
                        #             to_email  = assignee_email,
                        #             subject   = "Any update?",          # fixed subject
                        #             html_body = "qweqrtyuiop",       # resp in body
                        #             text_body = "asdfgh",
                        #         )
                        #     )
                        # print("resp------------",resp)
                        # if websocket:
                        #     await send_agent_response(websocket, resp)
                        # stop_task(task_id)

                    # 3) Joined-chat event → LLM crafts the welcome-back
                    # elif task["task_type"] == "user_joined_chat":
                    #     # First cancel any lingering “left” task
                    #     related = task["params"].get("task_id")
                    #     if related:
                    #         stop_task(related)

                    #     responses_dict[session_key]["conversation_history"].append({
                    #         "role": "system",
                    #         "content": {"type": "system: ", "chat": "User has rejoined the chat"},
                    #         "timestamp" : datetime.utcnow().isoformat()
                    #     })
                    #     resp = await parentAgent(websocket, responses_dict[session_key],user_id=None,wedding_session_id=None, watcher_event=True)
                    #     if websocket:
                    #         await send_agent_response(websocket, resp)
                    #     stop_task(task_id)

                    # 4) Fallback for anything else
                    else:
                        logger.info(
                            f"Unknown watcher task: {
                                task['task_type']} for user {uid}"
                        )
                        stop_task(task_id)

                    task["notified_parent"] = True

                    # ——— NEW: invite_watcher ———
                    if task["task_type"] == "invite_watcher":
                        # look up the pending invite by phone
                        inv = db["userOperations"].find_one(
                            {
                                "phone": params["phone"],
                                "bearer_token": {"$exists": True},
                                "user_id": None,  # still hasn’t joined
                            }
                        )

                        if not inv:
                            # invite record gone or already accepted → stop
                            # firing
                            stop_task(task_id)
                        else:
                            # still pending → nudge the bride
                            if websocket:
                                await send_notification(
                                    websocket,
                                    f"🎉 {
                                        inv['name']} still hasn’t joined your session.",
                                )
                            # reset timer so we can remind again after another
                            # interval
                            task["start_time"] = datetime.now()

                        # skip the “mark completed” step, leave task.active for
                        # next round
                        continue
                    # ——— end invite_watcher ———

                    # —— your existing branches start here —— #
                    if task["task_type"] == "handle_idle_event":
                        # … idle logic …
                        stop_task(task_id)

                    elif task["task_type"] == "user_left_chat":
                        # … left-chat logic …
                        stop_task(task_id)

                    elif task["task_type"] == "Remainder_after_3days":
                        # … 3-day-reminder logic …
                        stop_task(task_id)

                    elif task["task_type"] == "user_joined_chat":
                        # … joined-chat logic …
                        stop_task(task_id)

                    else:
                        logger.info(f"Unknown task_type: {task['task_type']}")
                        stop_task(task_id)

                    # after any non-invite watcher, mark it fired so it won’t
                    # repeat
                    task["notified_parent"] = True

                # fully cleaned-up tasks (completed + notified) get removed
                # here
                for tid in to_remove:
                    task_registry.pop(tid, None)

                await asyncio.sleep(30)

            # Clean up tasks marked completed but not yet notified
            elif task["status"] == "completed" and not task.get(
                "notified_parent", False
            ):
                await notify_parent_of_task_status(
                    task["user_id"], task_id, task["task_type"], completed=True
                )
                task["notified_parent"] = True
                to_remove.append(task_id)

        # Remove fully handled tasks from the registry
        for tid in to_remove:
            task_registry.pop(tid, None)

        await asyncio.sleep(30)


def process_recommendation_response(
    resp, bearer_token, wedding_session_id, segment=None
):
    """Process and convert UUID-based recommendation responses to actual recommendations"""

    if not isinstance(resp, dict):
        return resp

    content = resp.get("content", "")

    # Check if content contains recommendations_uuid format
    if isinstance(content, str) and "recommendations_uuid:" in content:
        # Extract values using regex
        uuid_match = re.search(r"recommendations_uuid:\s*([\w-]+)", content)

        if uuid_match:
            recommendation_uuid = uuid_match.group(1)

            # Get recommendations using the same logic as in
            # get_segment_chat_with_recommendations
            def get_recommendation_by_uuid(
                session_id: str, bearer_token: str, recommendation_uuid: str
            ):
                collection = db["chat_recommendations"]

                # Find user by bearer_token
                user_doc = db["userOperations"].find_one(
                    {"bearer_token": bearer_token})
                if not user_doc or "_id" not in user_doc:
                    return {"error": "Invalid bearer token"}

                user_obj_id = user_doc["_id"]

                # Find recommendations for this user/session
                doc = collection.find_one(
                    {
                        "user_id": user_obj_id,
                        "session_id": session_id,
                    }
                )

                if not doc:
                    return {
                        "error": "No recommendations found for this user/session."}

                recs = doc.get("recommendations", {})
                if not recs:
                    return {"error": "No recommendations found."}

                for vendor_cat, vendor_data in recs.items():
                    if (
                        isinstance(vendor_data, dict)
                        and recommendation_uuid in vendor_data
                    ):
                        uuid_data = vendor_data[recommendation_uuid]
                        # Clean the data to remove NaN/inf and convert ObjectId
                        # to str
                        safe_data = _clean_bson(uuid_data)
                        return {"recommendations": safe_data}

                return {"error": "Recommendation UUID not found."}

            # Get the actual recommendations
            recommendations_content = get_recommendation_by_uuid(
                session_id=wedding_session_id,
                bearer_token=bearer_token,
                recommendation_uuid=recommendation_uuid,
            )

            # Convert to the proper format
            return {
                "type": "recommendation_agent",
                "content": json.dumps({"Artifact": recommendations_content}),
                "segment": segment or resp.get("segment"),
            }

    return resp


def get_user_collection(bearer_token: str):
    """Get user-specific MongoDB collection based on bearer token."""
    collection_name = f"wedding_info_{bearer_token}"
    print(f"Accessing collection: {collection_name}")
    return db[collection_name]


def append_history(responses_dict, session_key, role, content):
    if isinstance(content, dict):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except Exception:
            content = str(content)
    responses_dict[session_key]["conversation_history"].append(
        {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
    )


def register_task(
    task_id: str,
    user_id: str,
    wedding_session_id: str,
    task_type: str,
    params: dict,
    timeout_minutes: int = 0,
):
    print(
        "Registering task",
        task_id,
        wedding_session_id,
        user_id,
        task_type,
        params,
        timeout_minutes,
    )
    """Register a new task dynamically with a timeout and notification flag."""
    task_registry[task_id] = {
        "wedding_session_id": wedding_session_id,
        "user_id": user_id,
        "task_type": task_type,
        "params": params,
        "status": "active",
        "start_time": datetime.now(),
        "end_time": None,
        "timeout_minutes": timeout_minutes,
        "notified_parent": False,  # To avoid duplicate notifications
    }
    print(
        f"Task {task_id} registered for user {user_id}, type: {task_type}, timeout: {timeout_minutes} min"
    )


def stop_task(task_id: str):
    if task_id in task_registry:
        task_registry[task_id]["status"] = "completed"
        task_registry[task_id]["end_time"] = datetime.now()
        print(f"Task {task_id} completed")
    else:
        print(f"Task {task_id} not found")


# Global dictionary to hold wedding info per user
wedding_info_global: Dict[str, WeddingInfo] = {}


@app.post("/auth/firebase", response_model=Token,
          openapi_extra={"security": []})
async def firebase_login(body: FirebaseLoginRequest):
    print("Received id_token:", body.id_token)
    try:
        decoded = firebase_auth.verify_id_token(body.id_token)
        print("Token verified successfully!")
        print("Decoded token payload:", json.dumps(decoded, indent=2))
    except Exception as e:
        print("Token verification failed:", str(e))
        raise HTTPException(401, f"Invalid Firebase ID token: {str(e)}")
    bearer_token = generate_bearer_token()

    user_data = {
        "uid": decoded.get("uid"),
        "email": decoded.get("email"),
        "email_verified": decoded.get("email_verified"),
        "bearer_token": bearer_token,
        "created_at": datetime.utcnow(),
        "name": decoded.get("name"),
        "picture": decoded.get("picture"),
        "provider": decoded.get("firebase", {}).get("sign_in_provider", "unknown"),
    }

    user_ops_col = db["userOperations"]
    existing_user = user_ops_col.find_one({"uid": user_data["uid"]})

    if existing_user and "bearer_token" in existing_user:
        bearer_token = existing_user["bearer_token"]
    else:
        bearer_token = generate_bearer_token()
        user_data["bearer_token"] = bearer_token
        user_ops_col.update_one(
            {"uid": user_data["uid"]}, {"$set": user_data}, upsert=True
        )

    return {"access_token": bearer_token, "token_type": "bearer"}


@app.post("/signup-login")
async def signup(body: EmailLoginRequest):
    email = body.email
    name = body.name
    print("Received Email:", email)
    print("Received Name:", name)

    sayyesai_db = client["SayYesAI"]
    user_ops_col = sayyesai_db["userOperations"]

    # Check if user with this email already exists
    existing_user = user_ops_col.find_one({"email": email})

    if existing_user:
        print("User already exists")
        print("Bearer Token:", existing_user["bearer_token"])

        # Update name only if provided and it's not a placeholder value
        if (
            name is not None
            and name.strip()
            and name.strip().lower() not in ["string", "name", "user", "test"]
        ):
            # Update the name in the database only if it's a meaningful name
            user_ops_col.update_one(
                {"email": email},
                {"$set": {"name": name, "updated_at": datetime.utcnow()}},
            )
            print(f"Updated name to: {name}")
        else:
            # If no meaningful name provided, check if user has existing name
            # in DB
            if (
                "name" not in existing_user
                or not existing_user["name"]
                or existing_user["name"] in ["string", "name", "user", "test"]
            ):
                # Only set default "name" if no meaningful name exists in DB
                user_ops_col.update_one(
                    {"email": email},
                    {"$set": {"name": "name", "updated_at": datetime.utcnow()}},
                )
                print("Set default name as no meaningful name existed in DB")

        # If user exists, return their existing bearer token
        return {
            "access_token": existing_user["bearer_token"], "token_type": "bearer"}
    else:
        print("User doesn't exist")
        # If user doesn't exist, create a new entry
        bearer_token = generate_bearer_token()
        user_data = {
            "email": email,
            "name": (
                name if name is not None else "name"
            ),  # Default to "name" only for new users
            "bearer_token": bearer_token,
            "created_at": datetime.utcnow(),
        }
        print("Bearer Token:", bearer_token)
        user_ops_col.insert_one(user_data)
        return {"access_token": bearer_token, "token_type": "bearer"}


@app.get("/AllUserWeddingParty", tags=["Wedding Operations"])
async def get_all_user_wedding_sessions(
    bearer_token: str = Header(..., description="Your existing bearer token")
):
    # 1) authenticate
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    user_id = user["_id"]

    # 2) find all sessions where you’re either the bride or a member/updater
    cursor = db["weddingSessionOperations"].find(
        {"$or": [{"users.bride": user_id}, {"users.user_id": user_id}]}
    )

    # 3) serialize
    sessions = []
    for raw in cursor:
        doc = dict(raw)
        oid = doc.pop("_id")
        doc["session_id"] = str(oid)
        # ensure no BSON types remain
        sessions.append(json.loads(json.dumps(doc, default=str)))

    if not sessions:
        raise HTTPException(
            status_code=404, detail="No wedding sessions found for this user"
        )

    return sessions


@app.post("/extract-name-llm/")
async def extract_name(data: Sentence):
    # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an intelligent assistant. The user will provide a sentence that may contain one or more names. "
                        "Your task is to extract and return the correct name of the person being referred to. "
                        "If the sentence contains multiple names, analyze carefully and return the name most likely referring to the speaker's own name. "
                        "Even if the name is short or lacks additional context, you should return it as the name. "
                        "Only return the name itself, with no explanation, punctuation, or additional text."
                    ),
                },
                {"role": "user", "content": data.text},
            ],
            temperature=0.2,
        )

        name = response.choices[0].message.content.strip()
        return name

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def send_blue_message(number: str, content: str):
    logger.info(f"Dummy SendBlue: sending to {number} content: {content}")
    await asyncio.sleep(0.1)
    # simulate that as soon as we "invite" them, they "join" shortly after
    # dummy_joined.add(number)
    return {"status": "sent", "number": number}


def _clean_bson(obj):
    if isinstance(obj, ObjectId):
        return str(obj)

    if isinstance(obj, Decimal128):
        dec = obj.to_decimal()
        return float(dec) if dec.is_finite() else None

    if isinstance(obj, float) and math.isnan(obj):
        return None

    if isinstance(obj, (dt.datetime, dt.date)):  # ← now dt.*
        return obj.isoformat()

    # ----------------------------------------------------------------
    # containers
    # ----------------------------------------------------------------
    if isinstance(obj, list):
        return [_clean_bson(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean_bson(v) for k, v in obj.items()}
    return obj


def id_variants(value: str) -> Dict[str, Any]:

    opts = [value]
    try:
        opts.append(ObjectId(value))
    except (bson_errors.InvalidId, TypeError):
        # Not a valid ObjectId representation – just ignore
        pass
    return {"$in": opts}


def _clean_bson_(obj):
    if isinstance(obj, ObjectId):
        return str(obj)  # Convert ObjectId to string

    if isinstance(obj, Decimal128):
        dec = obj.to_decimal()
        return float(dec) if dec.is_finite() else None

    if isinstance(obj, float) and math.isnan(obj):
        return None

    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()

    # Recursively clean containers
    if isinstance(obj, list):
        return [_clean_bson(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clean_bson(v) for k, v in obj.items()}
    return obj


def ensure_json_serializable(obj):
    """Convert ObjectId and datetime to string for JSON serialization"""
    from datetime import datetime

    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    else:
        return obj


@app.get("/in-app-notifications", tags=["Notifications"])
async def get_in_app_notifications(
    bearer_token: str = Header(...,
                               description="Bearer token for authentication"),
    wedding_session_id: str = Query(..., description="Wedding session ID"),
    unread_only: bool = Query(
        False, description="Fetch only unread notifications"),
    limit: int = Query(20, description="Maximum notifications to return"),
):
    """
    Get in-app notifications for the authenticated user.
    Bridesmaids call this to check for new notifications.
    """
    # Authenticate user
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = str(user["_id"])

    # Build query
    query = {"wedding_session_id": wedding_session_id, "target_users": user_id}

    if unread_only:
        query["is_read"] = False

    # Fetch notifications
    notifications = list(
        db["in_app_notifications"].find(query).sort(
            "created_at", -1).limit(limit)
    )

    # Filter: for task reminders, only include if the task is
    # yesterday/today/tomorrow
    today_utc = datetime.utcnow().date()
    # Fallback if IST not available
    today_ist = today_utc
    allowed_dates = {
        (today_ist - timedelta(days=1)).isoformat(),
        today_ist.isoformat(),
        (today_ist + timedelta(days=1)).isoformat(),
    }

    def _include_notification(doc: dict) -> bool:
        nd = doc.get("notification_data", {}) or {}
        ntype = nd.get("type")
        if ntype == "task_reminder":
            due_str = nd.get("due_date") or nd.get("dueDate")
            if isinstance(due_str, str) and due_str in allowed_dates:
                return True
            created = doc.get("created_at")
            if isinstance(created, datetime):
                # Approximate IST by adding +5:30 to UTC naive timestamps
                ist_date = (created + timedelta(hours=5, minutes=30)).date()
                if ist_date.isoformat() in allowed_dates:
                    return True
            return False
        return True

    notifications = [n for n in notifications if _include_notification(n)]

    cleaned_notifications = ensure_json_serializable(notifications)

    return {"notifications": cleaned_notifications,
            "count": len(cleaned_notifications)}


@app.get("/in-app-notifications/unread-count", tags=["Notifications"])
async def get_unread_notification_count(
    bearer_token: str = Header(...,
                               description="Bearer token for authentication"),
    wedding_session_id: str = Query(..., description="Wedding session ID"),
):
    """
    Get count of unread notifications. Use this for notification badges.
    """
    # Authenticate user
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = str(user["_id"])

    # Count unread notifications
    count = db["in_app_notifications"].count_documents(
        {
            "wedding_session_id": wedding_session_id,
            "target_users": {"$in": [user_id]},
            # User has NOT read this notification
            "read_by.user_id": {"$ne": user_id},
        }
    )

    return {"unread_count": count}


@app.post("/in-app-notifications/mark-read", tags=["Notifications"])
async def mark_notifications_read(
    bearer_token: str = Header(...,
                               description="Bearer token for authentication"),
    notification_ids: List[str] = Body(
        ..., description="List of notification IDs to mark as read"
    ),
):
    """
    Mark specific notifications as read.
    """
    # Authenticate user
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = str(user["_id"])

    # Convert to ObjectIds
    try:
        object_ids = [ObjectId(nid) for nid in notification_ids]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid notification IDs")

    # Update notifications
    result = db["in_app_notifications"].update_many(
        {
            "_id": {"$in": object_ids},
            "target_users": {"$in": [user_id]},
            # Only if user hasn't already read it
            "read_by.user_id": {"$ne": user_id},
        },
        {"$push": {"read_by": {"user_id": user_id, "read_at": datetime.utcnow()}}},
    )

    return {
        "message": f"Marked {result.modified_count} notifications as read for this user",
        "modified_count": result.modified_count,
    }


def clean_doc(doc: dict) -> dict:
    """Convert ObjectId and datetime to strings recursively."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, dict):
            out[k] = clean_doc(v)
        elif isinstance(v, list):
            out[k] = [
                (
                    clean_doc(x)
                    if isinstance(x, dict)
                    else (str(x) if isinstance(x, (ObjectId, datetime)) else x)
                )
                for x in v
            ]
        else:
            out[k] = v
    return out


@app.get("/notifications", tags=["Notifications"])
def get_notifications(
    bearer_token: str = Header(..., description="Bearer token of the user"),
    wedding_session_id: str = Header(..., description="Wedding session ID"),
):
    """
    Fetch notifications for the authenticated user within a wedding session.
    Steps:
      1. Authenticate user by bearer_token.
      2. Validate that user is part of the given wedding_session_id.
      3. Find all notifications for that session.
      4. Return only those where user_id is in target_users.
    """

    # Step 1: Authenticate user
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user or "_id" not in user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    user_id = str(user["_id"])

    # Step 2: Validate session access
    try:
        _auth_session(
            bearer_token=bearer_token,
            wedding_session_id=wedding_session_id)
    except Exception:
        raise HTTPException(
            status_code=403, detail="Not authorized for this wedding session"
        )

    # Step 3: Fetch all notifications for the session
    notifications = list(
        db["in_app_notifications"].find(
            {"wedding_session_id": wedding_session_id})
    )
    print("*" * 60)
    print(
        "in getting notifocation end point , number of notifications found:",
        len(notifications),
    )

    # check weather user_id is in target_users (pariticipant of session ) of
    # each notification
    user_notifications = [
        n for n in notifications if user_id in n.get("target_users", [])
    ]

    # Clean ObjectIds for JSON serialization
    safe_notifications = ensure_json_serializable(user_notifications)

    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "count": len(safe_notifications),
            "data": safe_notifications,
        },
    )


@app.post("/progress/manual-update", tags=["User Operations"])
async def manual_update_progress(
    bearer_token: str = Header(...,
                               description="Bearer token for user authentication"),
    wedding_session_id: str = Header(
        ..., description="Wedding session ID to update progress for"
    ),
    category: str = Header(
        ..., description="Category to update (e.g. 'venues', 'djs')"
    ),
    points: int = Header(...,
                         description="Number of points to add to the category"),
    type: str = Header(
        None,
        description="Optional type to set in progress_flags (e.g. 'searched', 'saved')",
    ),
):
    db = get_db()
    collection = db["progress"]

    query = {"bearer_token": bearer_token,
             "wedding_session_id": wedding_session_id}

    update_query = {"$inc": {category: points}}

    if type:
        update_query["$set"] = {f"progress_flags.{category}.{type}": True}

    result = collection.update_one(query, update_query, upsert=True)

    if result.modified_count > 0 or result.upserted_id:
        return JSONResponse(
            status_code=200,
            content={
                "message": f"{points} points added to '{category}'.",
                "type_set": type or None,
            },
        )
    else:
        raise HTTPException(status_code=400, detail="Progress update failed.")


@app.get("/onboarding-status", tags=["User Operations"])
async def get_onboarding_progress_endpoint(
    bearer_token: str = Header(..., alias="bearer-token"),
    wedding_session_id: str = Header(..., alias="wedding-session-id"),
):
    """
    Get the current onboarding progress status for a user.

    Returns:
    - progress_percentage: 0, 50, or 100
    - categories_saved: List of categories where recommendations were saved
    - onboarding_complete: Boolean indicating if onboarding is complete
    """
    try:
        # Validate session (reuse your existing auth logic)
        _auth_session(bearer_token, wedding_session_id)

        result = await get_onboarding_status(bearer_token, wedding_session_id)

        return {"success": True, "data": result}

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error getting onboarding status: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving onboarding status: {str(e)}"
        )


@app.get("/progress", tags=["User Operations"])
async def get_progress(
    bearer_token: str = Header(...,
                               description="Bearer token for user authentication"),
    wedding_session_id: str = Query(
        ..., description="Wedding session ID to fetch progress"
    ),
):
    """
    Get the progress details from the 'progress' collection for a given bearer token and wedding session ID.
    """

    if not bearer_token or not wedding_session_id:
        raise HTTPException(
            status_code=400,
            detail="Both 'bearer_token' and 'wedding_session_id' are required.",
        )

    # 1. Query the progress collection by bearer_token and wedding_session_id
    progress_data = db["progress"].find_one(
        {"bearer_token": bearer_token, "wedding_session_id": wedding_session_id}
    )

    if not progress_data:
        raise HTTPException(status_code=404, detail="Progress data not found.")

    # 2. Clean ObjectId fields
    progress_data = _clean_bson(progress_data)

    return JSONResponse(content=progress_data, status_code=200)


# ------------------------------------------------------------------
@app.get("/mood-board-all recommendations", tags=["Wedding Recommendations"])
async def get_mood_board(
    *,
    user_id: Optional[str] = Query(None, description="User ID"),
    bearer_token: Optional[str] = Header(None, description="Bearer Token"),
    session_id: str = Query(..., description="Wedding session ID"),
):
    if bearer_token:
        user = db["userOperations"].find_one({"bearer_token": bearer_token})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        user_id = str(user["_id"])
    elif user_id:
        user_id = user_id
    else:
        raise HTTPException(
            status_code=400, detail="Either user_id or bearer_token must be provided"
        )

    col = db["mood_board"]

    raw = col.find_one(
        {"user_id": id_variants(user_id),
         "session_id": id_variants(session_id)},
        {"_id": 0},
    )
    if not raw:
        raise HTTPException(404, "No mood-board found for that user & session")

    return JSONResponse(content=_clean_bson(raw), status_code=200)


# ------------------------------------------------------------------
# 2.  /mood-board/categories             (list available categories)
# ------------------------------------------------------------------
@app.get("/mood-board/categories", tags=["Wedding Recommendations"])
async def list_mood_board_categories(
    *,
    user_id: Optional[str] = Query(None, description="User ID"),
    bearer_token: Optional[str] = Header(None, description="Bearer Token"),
    session_id: str = Query(..., description="Planning session ID"),
):
    if bearer_token:
        user = db["userOperations"].find_one({"bearer_token": bearer_token})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        user_id = str(user["_id"])
    elif user_id:
        user_id = user_id
    else:
        raise HTTPException(
            status_code=400, detail="Either user_id or bearer_token must be provided"
        )

    col = db["mood_board"]
    doc = col.find_one(
        {"user_id": id_variants(user_id),
         "session_id": id_variants(session_id)},
        {"_id": 0, "recommendations": 1},
    )
    if not doc:
        raise HTTPException(404, "No mood-board found for that user & session")

    return {"categories": list(_clean_bson(
        doc).get("recommendations", {}).keys())}


# ------------------------------------------------------------------
# 3.  /mood-board/{category}             (get one category)
# ------------------------------------------------------------------
@app.get("/mood-board/{category}", tags=["Wedding Recommendations"])
async def get_mood_board_category(
    *,
    category: str,
    user_id: Optional[str] = Query(None, description="User ID"),
    session_id: str = Query(..., description="Planning session ID"),
    bearer_token: Optional[str] = Header(None, description="Bearer Token"),
):
    if bearer_token:
        user = db["userOperations"].find_one({"bearer_token": bearer_token})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        user_id = str(user["_id"])
    elif user_id:
        user_id = user_id
    else:
        raise HTTPException(
            status_code=400, detail="Either user_id or bearer_token must be provided"
        )

    col = db["mood_board"]
    doc = col.find_one(
        {"user_id": id_variants(user_id),
         "session_id": id_variants(session_id)},
        {"_id": 0, f"recommendations.{category}": 1},
    )
    if not doc:
        raise HTTPException(404, "No mood-board found for that user & session")

    data = _clean_bson(doc).get("recommendations", {}).get(category)
    if data is None:
        raise HTTPException(
            404, f"No category '{category}' in that mood-board")

    return JSONResponse(content=data, status_code=200)


@app.delete("/mood-board/recommendation", tags=["Wedding Recommendations"])
async def delete_recommendation(
    *,
    bearer_token: str = Header(..., description="Bearer Token"),
    session_id: str = Header(..., description="Wedding session ID"),
    title: str = Body(...,
                      description="Title of the recommendation to delete"),
    category: Optional[str] = Body(
        default=None, description="Optional category to search within"
    ),
):
    # Authentication
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    user_id = str(user["_id"])

    col = db["mood_board"]

    # Check if document exists
    doc = col.find_one(
        {"user_id": id_variants(user_id),
         "session_id": id_variants(session_id)},
        {"_id": 0, "recommendations": 1},
    )
    if not doc:
        raise HTTPException(200, "No mood-board found for that user & session")

    recommendations = _clean_bson(doc).get("recommendations", {})

    if category and category.strip() and category != "string":
        # Search in specific category only
        if category not in recommendations:
            raise HTTPException(
                200, f"No category '{category}' found in mood-board")

        # Check if recommendation exists before deletion
        category_items = recommendations.get(category, [])
        recommendation_exists = any(
            item.get("title") == title for item in category_items
        )

        if not recommendation_exists:
            raise HTTPException(
                404,
                f"No recommendation with title '{title}' found in category '{category}'",
            )

        # If this is the only item in the category, remove the entire category
        if len(category_items) == 1:
            result = col.update_one(
                {
                    "user_id": id_variants(user_id),
                    "session_id": id_variants(session_id),
                },
                {"$unset": {f"recommendations.{category}": ""}},
            )
            # Also remove corresponding post(s) from voting_collection
            try:
                voting_filter = {
                    "bearer_token": bearer_token,
                    "session_id": session_id,
                    "category": category,
                    "post_type": "recommendation",
                    "post.title": title,
                }
                db["voting_collection"].delete_many(voting_filter)
            except Exception as e:
                print(f"Warning: failed to delete from voting_collection: {e}")
            return JSONResponse(
                content={
                    "message": f"Recommendation '{title}' deleted successfully from category '{category}'. Category removed as it was empty."
                },
                status_code=200,
            )
        else:
            # Remove just the recommendation
            result = col.update_one(
                {
                    "user_id": id_variants(user_id),
                    "session_id": id_variants(session_id),
                },
                {"$pull": {f"recommendations.{category}": {"title": title}}},
            )
            # Also remove corresponding post(s) from voting_collection
            try:
                voting_filter = {
                    "bearer_token": bearer_token,
                    "session_id": session_id,
                    "category": category,
                    "post_type": "recommendation",
                    "post.title": title,
                }
                db["voting_collection"].delete_many(voting_filter)
            except Exception as e:
                print(f"Warning: failed to delete from voting_collection: {e}")
            return JSONResponse(
                content={
                    "message": f"Recommendation '{title}' deleted successfully from category '{category}'"
                },
                status_code=200,
            )

    else:
        # Search in all categories
        found_category = None
        found_item = None

        for cat_name in recommendations.keys():
            result = col.update_one(
                {
                    "user_id": id_variants(user_id),
                    "session_id": id_variants(session_id),
                },
                {"$pull": {f"recommendations.{cat_name}": {"title": title}}},
            )

            if result.modified_count > 0:
                # Also remove corresponding post(s) from voting_collection for
                # this category
                try:
                    voting_filter = {
                        "bearer_token": bearer_token,
                        "session_id": session_id,
                        "category": cat_name,
                        "post_type": "recommendation",
                        "post.title": title,
                    }
                    db["voting_collection"].delete_many(voting_filter)
                except Exception as e:
                    print(
                        f"Warning: failed to delete from voting_collection: {e}")
                found = True
                deleted_from_category = cat_name
                break  # Stop after finding and deleting the first match

        if not found:
            raise HTTPException(
                200, f"No recommendation with title '{title}' found in any category"
            )

        return JSONResponse(
            content={
                "message": f"Recommendation '{title}' deleted successfully from category '{deleted_from_category}'"
            },
            status_code=200,
        )

    # 1) convert ObjectId → str  (jsonable_encoder does that)
    # 2) replace NaN / ±inf with None so json.dumps won’t choke
    # ------------------------------------------------------------------
    def cleanse(val):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        if isinstance(val, dict):
            return {k: cleanse(v) for k, v in val.items()}
        if isinstance(val, list):
            return [cleanse(v) for v in val]
        return val

    safe_doc = cleanse(jsonable_encoder(raw, custom_encoder={ObjectId: str}))

    return JSONResponse(content=safe_doc, status_code=200)


# @app.get("/mood-board/categories", tags=["Wedding Recommendations"])
# async def list_mood_board_categories(
#     *,
# user_id: Optional[str] = Query(None, description="User ID"),
#     bearer_token: Optional[str] = Header(None, description="Bearer Token"),

#     session_id: str = Query(..., description="Planning session ID"),
# ):
#     if bearer_token:
#         user = db["userOperations"].find_one({"bearer_token": bearer_token})
#         if not user:
#             raise HTTPException(status_code=401, detail="Invalid bearer token")
#         user_id = str(user["_id"])
#     elif user_id:
#         user_id = user_id
#     else:
#         raise HTTPException(status_code=400, detail="Either user_id or bearer_token must be provided")


#     col = db["mood_board"]
#     doc = col.find_one({"user_id": user_id, "session_id": session_id},
#                        {"_id": 0, "recommendations": 1})
#     if not doc:
#         raise HTTPException(status_code=404,
# detail="No mood-board found for that user & session")

#     recs = doc.get("recommendations", {})
#     return {"categories": list(recs.keys())}


# @app.get("/mood-board/{category}", tags=["Wedding Recommendations"])
# async def get_mood_board_category(
#     *,
#     category: str,
# user_id: Optional[str] = Query(None, description="User ID"),
#     bearer_token: Optional[str] = Header(None, description="Bearer Token"),
#     session_id: str = Query(..., description="Planning session ID"),
# ):
#     if bearer_token:
#         user = db["userOperations"].find_one({"bearer_token": bearer_token})
#         if not user:
#             raise HTTPException(status_code=401, detail="Invalid bearer token")
#         user_id = str(user["_id"])
#     elif user_id:
#         user_id = user_id
#     else:
#         raise HTTPException(status_code=400, detail="Either user_id or bearer_token must be provided")

#     col = db["mood_board"]
#     doc = col.find_one({"user_id": user_id, "session_id": session_id},
#                        {"_id": 0, f"recommendations.{category}": 1})
#     if not doc:
#         raise HTTPException(status_code=404,
# detail="No mood-board found for that user & session")

#     data = doc.get("recommendations", {}).get(category)
#     if data is None:
#         raise HTTPException(status_code=404,
# detail=f"No category '{category}' in that mood-board")

#     # Clean BSON / NaN so JSONResponse doesn’t choke
#     safe = _clean_bson(jsonable_encoder(data))
#     return JSONResponse(content=safe, status_code=200)


@app.post("/wedding-date", tags=["Wedding Operations"])
async def set_or_update_wedding_date(
    wedding_date: str = Header(...,
                               description="Wedding date in DD-MM-YYYY format"),
    bearer_token: str = Header(..., description="Your existing bearer token"),
    wedding_session_id: str = Header(..., description="Wedding session ID"),
):
    # Step 1: Validate the wedding date format (DD-MM-YYYY)
    try:
        # Try to parse the wedding date and validate the format
        datetime.strptime(wedding_date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Please use DD-MM-YYYY format."
        )

    # Step 2: Find the user document using the bearer token
    user_collection = db["userOperations"]
    user_document = user_collection.find_one({"bearer_token": bearer_token})

    if not user_document:
        raise HTTPException(status_code=404, detail="User not found")

    # Retrieve the user ID from the document
    user_id = user_document["_id"]

    # Step 3: Find the wedding session that matches both the user ID and the
    # wedding session ID
    wedding_collection = db["weddingSessionOperations"]
    session = wedding_collection.find_one(
        {"_id": ObjectId(wedding_session_id), "users.bride": user_id}
    )

    if not session:
        raise HTTPException(
            status_code=404, detail="Wedding session not found for this user"
        )

    # Step 4: Update the wedding date
    result = wedding_collection.update_one(
        {"_id": session["_id"]}, {"$set": {"wedding_date": wedding_date}}
    )

    if result.modified_count:
        return {"message": "Wedding date updated successfully"}
    else:
        return {"message": "Wedding date was already up to date"}


@app.get("/wedding-date/{session_id}", tags=["Wedding Operations"])
async def get_wedding_date_by_bearer_token(
    bearer_token: str = Header(..., description="Your existing bearer token"),
    session_id: str = Path(..., description="Wedding session ID"),
):
    # Step 1: Search for the user based on bearer token
    user_operations_collection = db["userOperations"]
    user_doc = user_operations_collection.find_one(
        {"bearer_token": bearer_token})

    if not user_doc:
        raise HTTPException(
            status_code=404, detail="User not found with this bearer token"
        )

    # Step 2: Extract the user_id from the document
    user_id = user_doc.get("_id")

    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="User document is missing _id")

    # Step 3: Search for the wedding session using both user_id and session_id
    wedding_session_collection = db["weddingSessionOperations"]
    wedding_session = wedding_session_collection.find_one(
        {
            "_id": ObjectId(session_id),
            "$or": [{"users.bride": user_id}, {"users.participants": user_id}],
        }
    )

    if not wedding_session:
        raise HTTPException(
            status_code=404, detail="Wedding session not found for this user"
        )

    # Step 4: Return the wedding date
    return {"wedding_date": wedding_session.get("wedding_date", "Not set")}


# @app.post("/wedding-tasks", tags=["Wedding Task"])
# async def manage_wedding_tasks(data: TaskOperationInput,
#     bearer_token: str = Header(..., description="Your existing bearer token"),
#     wedding_session_id: str = Header(..., description="Wedding session ID")
# ):
#     """
#     Manage wedding tasks assigned to specific dates.

#     You can use this endpoint to:
#     - use "add" : Add a task to a specific date
#     - use "edit" : Edit (rename) a task on a date. Give old task in 'task' field and new task in 'new_task' field
#     - use "delete" : "Delete a specific task from a date

#     """
#     # Step 1: Authenticate the user using bearer token
#     user = db["userOperations"].find_one({"bearer_token": bearer_token})
#     if not user:
#         raise HTTPException(status_code=401, detail="Invalid bearer token")
#     user_id = user["_id"]

#     # Step 2: Fetch the wedding session by matching user_id and wedding_session_id
#     wedding_session = db["weddingSessionOperations"].find_one({
#         "_id": ObjectId(wedding_session_id),
#         "users.bride": user_id
#     })

#     if not wedding_session:
#         raise HTTPException(status_code=404, detail="Wedding session not found")

#     # -- Add task --------------------------------------------------------
#     if data.action == "add":
#         if not data.task:
#             raise HTTPException(status_code=400, detail="'task' object required")

#         # 1️⃣ make a unique, URL-safe ID for this task
# task_uid = str(uuid4())          # e.g.
# "b8e9ce27-0d31-48c0-9059-7b649f3e7184"

#         # 2️⃣ build the document to insert
#         task_doc = data.task.to_dict()
#         task_doc.update(
#             {
#                 "task_id": task_uid,          # our own public identifier
#                 "bearer_token": bearer_token,
#                 "wedding_session_id": wedding_session_id,
#             }
#         )

#         # 3️⃣ insert (let Mongo generate its own _id silently)
#         db["wedding_tasks"].insert_one(task_doc)

#         # 4️⃣ link that task back to the session (push the *uuid*)
#         db["weddingSessionOperations"].update_one(
#             {"_id": ObjectId(wedding_session_id)},
#             {"$push": {f"wedding_tasks.{data.task.assignedDate}": task_uid}},
#         )

#     # ------------------------------------------------------------------
#     # -- Edit task -----------------------------------------------------
#     # ------------------------------------------------------------------
#     elif data.action == "edit":
#         if not (data.task and data.task_id):
#             raise HTTPException(status_code=400, detail="'task' and 'task_id' required")

#         db["wedding_tasks"].update_one(
#             {"task_id": data.task_id},          # look-up by our uuid
#             {"$set": data.task.to_dict()},
#         )

#     # ------------------------------------------------------------------
#     # -- Delete task ---------------------------------------------------
#     # ------------------------------------------------------------------
#     elif data.action == "delete":
#         if not data.task_id:
#             raise HTTPException(status_code=400, detail="'task_id' required")

#         db["wedding_tasks"].delete_one({"task_id": data.task_id})

#         # remove that uuid from **every** date bucket
#         db["weddingSessionOperations"].update_many(
#             {"_id": ObjectId(wedding_session_id)},
#             {"$pull": {"wedding_tasks.$[].": data.task_id}},
#         )
#     else:
#         raise HTTPException(status_code=400, detail="Unsupported action")

#     return {"message": f"Action '{data.action}' completed"}


def _auth_session(bearer_token: str, wedding_session_id: str):
    # 1. Lookup user by bearer_token
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = str(user["_id"])

    try:
        session_oid = ObjectId(wedding_session_id)
    except BaseException:
        raise HTTPException(status_code=400, detail="Invalid session ID")

    # 2. Fetch the wedding session document
    session = db["weddingSessionOperations"].find_one({"_id": session_oid})
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Wedding session not found")

    users_block = session.get("users", {})
    bride_id = str(users_block.get("bride"))
    participants_dict = users_block.get("participants", {})

    # 3. Check if user is either the bride or one of the participants
    is_participant = user_id in participants_dict.values()

    if user_id != bride_id and not is_participant:
        raise HTTPException(
            status_code=403, detail="You are not authorized for this session"
        )

    # ✅ Auth success


@app.post("/wedding-tasks", tags=["Wedding Task"])
async def add_wedding_task(
    bearer_token: str = Header(..., description="Bearer token of the bride"),
    wedding_session_id: str = Header(...,
                                     description="ID of the wedding session"),
    task_name: str = Header(..., description="Name of the wedding task"),
    assignee_id: Optional[str] = Header(
        None, alias="assignee-id", description="Bearer token of the bridesmaid/assignee"
    ),
    task_description: Optional[str] = Header(
        None, alias="task-description", description="Detailed description of the task"
    ),
    task_status: Optional[str] = Header(
        "pending", alias="task-status", description="Current status of the task"
    ),
    assigned_date: Optional[str] = Header(
        None, alias="assigned-date", description="Date when the task was assigned"
    ),
    due_date: Optional[str] = Header(
        None, alias="due-date", description="Deadline for completing the task"
    ),
):
    try:
        print(f"=== TASK CREATION DEBUG ===")
        print(f"bearer_token: {bearer_token}")
        print(f"assignee_id (bearer token): {assignee_id}")
        print(f"task_name: {task_name}")

        # Step 1: Validate that the caller is authorized for this session
        _auth_session(bearer_token, wedding_session_id)

        # Step 2: If assignee_id is provided, validate it's a valid bearer
        # token
        if assignee_id:
            print(f"Validating assignee bearer token: {assignee_id}")
            user_doc = db["userOperations"].find_one(
                {"bearer_token": assignee_id})
            if not user_doc:
                raise HTTPException(
                    status_code=400, detail="Invalid assignee bearer token"
                )

            print(f"Found assignee user: {user_doc.get('name', 'Unknown')}")
            assignee_user_id = str(user_doc["_id"])

            # Step 3: Ensure assignee is part of this wedding session
            print(f"Validating assignee is part of wedding session...")
            wedding_session = db["weddingSessionOperations"].find_one(
                {"_id": ObjectId(wedding_session_id)}
            )
            if not wedding_session:
                raise HTTPException(
                    status_code=404,
                    detail="Wedding session not found")

            bride_id = str(wedding_session["users"].get("bride"))
            participant_ids = [
                str(uid)
                for uid in wedding_session["users"].get("participants", {}).values()
            ]

            print(f"Bride ID: {bride_id}")
            print(f"Participant IDs: {participant_ids}")
            print(f"Assignee User ID: {assignee_user_id}")

            if assignee_user_id not in ([bride_id] + participant_ids):
                raise HTTPException(
                    status_code=401,
                    detail="Assignee is not part of this wedding session",
                )
            print(f"✓ Assignee validated successfully")

        # Step 4: Create and insert the task
        task_uid = str(uuid4())
        task_doc = {
            "task_id": task_uid,
            "taskName": task_name,
            "taskStatus": task_status or "pending",
            "bearer_token": bearer_token,  # Bride's bearer token
            "wedding_session_id": wedding_session_id,
            "createdAt": datetime.utcnow(),
        }

        # Add assignee bearer token directly
        if assignee_id:
            # Store assignee's bearer token
            task_doc["assignee_id"] = assignee_id

        # Add optional fields
        if task_description:
            task_doc["taskDescription"] = task_description
        if assigned_date:
            task_doc["assignedDate"] = assigned_date
        if due_date:
            task_doc["dueDate"] = due_date

        print(f"Task document to insert: {task_doc}")
        db["wedding_tasks"].insert_one(task_doc)

        # Step 5: Link task to session if assigned_date is given
        if assigned_date:
            db["weddingSessionOperations"].update_one(
                {"_id": ObjectId(wedding_session_id)},
                {"$push": {f"wedding_tasks.{assigned_date}": task_uid}},
            )

        print(f"✓ Task created successfully with ID: {task_uid}")
        return {"message": "Task added", "task_id": task_uid}

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR in task creation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add task: {
                str(e)}")


@app.put("/wedding-tasks", tags=["Wedding Task"])
async def rename_wedding_task(
    bearer_token: str = Header(..., alias="bearerToken"),
    wedding_session_id: str = Header(..., alias="weddingSessionId"),
    task_id: str = Header(..., alias="taskId"),
    task_name: str = Header(..., alias="taskName"),  # new name
):
    _auth_session(bearer_token, wedding_session_id)

    result = db["wedding_tasks"].update_one(
        {"task_id": task_id},
        {"$set": {"taskName": task_name}},
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"message": "Task name updated",
            "task_id": task_id, "newName": task_name}


@app.put("/wedding-tasks/update", tags=["Wedding Task"])
async def update_wedding_task(
    bearer_token: str = Header(..., alias="bearerToken"),
    wedding_session_id: str = Header(..., alias="weddingSessionId"),
    task_name: Optional[str] = Header(
        None, alias="taskName"
    ),  # Optional: task name to find the task
    task_id: Optional[str] = Header(
        None, alias="taskId"
    ),  # Optional: task ID to find the task
    assignee: Optional[str] = Header(None),  # optional: new assignee
    assignee_id: Optional[str] = Header(
        None, alias="assigneeId"
    ),  # updated header name
    task_description: Optional[str] = Header(
        None, alias="task-description"
    ),  # optional: new task description
    task_status: Optional[str] = Header(
        None, alias="task-status"
    ),  # optional: new task status
    assigned_date: Optional[str] = Header(
        None, alias="assigned-date"
    ),  # optional: new assigned date
    due_date: Optional[str] = Header(
        None, alias="due-date"),  # optional: new due date
):
    # Authenticate the session with bearer_token and wedding_session_id
    _auth_session(bearer_token, wedding_session_id)

    # Find the task based on either taskName or taskId (one of these should be
    # provided)
    if task_name:
        task = db["wedding_tasks"].find_one({"taskName": task_name})
    elif task_id:
        task = db["wedding_tasks"].find_one({"task_id": task_id})
    else:
        raise HTTPException(
            status_code=400, detail="Either taskName or taskId must be provided"
        )

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Prepare the fields to update
    update_fields = {}

    if assignee is not None:
        update_fields["assignee"] = assignee
    if assignee_id is not None:
        update_fields["id"] = assignee_id
    if task_description is not None:
        update_fields["taskDescription"] = task_description
    if task_status is not None:
        update_fields["taskStatus"] = task_status
    if assigned_date is not None:
        update_fields["assignedDate"] = assigned_date
    if due_date is not None:
        update_fields["dueDate"] = due_date

    # Perform the update in the database
    result = db["wedding_tasks"].update_one(
        {"_id": task["_id"]},  # Update the task using the _id
        {"$set": update_fields},
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")

    # Prepare the updated task (using the modified fields)
    # Merge original task and updated fields
    updated_task = {**task, **update_fields}

    # Convert ObjectId fields to string for serializability
    updated_task["_id"] = str(updated_task["_id"])
    if "wedding_tasks" in updated_task:
        updated_task["wedding_tasks"] = [
            str(wedding_task) for wedding_task in updated_task["wedding_tasks"]
        ]

    # Return the updated task details
    return {
        "message": "Task details updated successfully",
        "updated_task": updated_task,
    }


def _auth_bride_only(bearer_token: str, wedding_session_id: str):
    """
    Authenticate and authorize - ONLY BRIDE can perform this action
    """
    print(f"=== BRIDE-ONLY AUTH DEBUG ===")
    print(f"Bearer token: {bearer_token}")
    print(f"Wedding session ID: {wedding_session_id}")

    # 1. Authenticate the user
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        print(f"❌ User not found with bearer token")
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = str(user["_id"])
    print(f"✓ Found user: {user.get('name', 'Unknown')} (ID: {user_id})")

    # 2. Validate session ID
    try:
        session_oid = ObjectId(wedding_session_id)
    except BaseException:
        print(f"❌ Invalid session ID format")
        raise HTTPException(status_code=400, detail="Invalid session ID")

    # 3. Get the wedding session
    session = db["weddingSessionOperations"].find_one({"_id": session_oid})
    if not session:
        print(f"❌ Wedding session not found")
        raise HTTPException(
            status_code=404,
            detail="Wedding session not found")

    # 4. Check if user is the bride (NOT a participant)
    bride_id = str(session.get("users", {}).get("bride"))
    print(f"Bride ID: {bride_id}")
    print(f"User ID: {user_id}")

    if user_id != bride_id:
        print(f"❌ User is not the bride - access denied")
        raise HTTPException(status_code=403,
                            detail="Only the bride can delete tasks")

    print(f"✓ Bride authorization successful")
    return True


@app.delete("/wedding-tasks", tags=["Wedding Task"])
async def delete_wedding_task(
    bearer_token: str = Header(..., alias="bearerToken"),
    wedding_session_id: str = Header(..., alias="weddingSessionId"),
    task_id: str = Header(..., alias="taskId"),
):
    print(f"=== DELETE TASK DEBUG ===")
    print(f"Task ID to delete: {task_id}")

    # Use bride-only authentication
    _auth_bride_only(bearer_token, wedding_session_id)

    # Find the task
    task = db["wedding_tasks"].find_one({"task_id": task_id})
    if not task:
        print(f" Task not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    print(f" Found task: {task.get('taskName')}")

    # Additional check: Verify task belongs to this wedding session
    if task.get("wedding_session_id") != wedding_session_id:
        print(f" Task belongs to different session")
        raise HTTPException(
            status_code=403, detail="Task does not belong to this wedding session"
        )

    # Remove the task document
    delete_result = db["wedding_tasks"].delete_one({"task_id": task_id})
    if delete_result.deleted_count == 0:
        print(f" Failed to delete task from database")
        raise HTTPException(status_code=500, detail="Failed to delete task")

    print(f" Task deleted from database")

    # Unlink its id from the correct date bucket (only if assignedDate exists)
    assigned_date = task.get("assignedDate")
    if assigned_date:
        update_result = db["weddingSessionOperations"].update_one(
            {"_id": ObjectId(wedding_session_id)},
            {"$pull": {f"wedding_tasks.{assigned_date}": task_id}},
        )
        print(f" Unlinked task from session date bucket: {assigned_date}")
    else:
        print(" No assigned date, skipping session unlinking")

    print(f" Task deletion completed successfully")
    return {"message": "Task deleted", "task_id": task_id}


@app.get("/wedding-tasks", tags=["Wedding Task"])
async def get_wedding_tasks(
    bearer_token: str = Header(..., alias="bearerToken"),
    wedding_session_id: str = Header(..., alias="weddingSessionId"),
    assignee: str | None = Query(
        None, description="Filter by assignee bearer token (optional)"
    ),
    status: str | None = Query(
        None,
        pattern="^(pending|doing|done)$",
        description="Filter by status (optional)",
    ),
    assigned_date: str | None = Query(
        None, description="Filter by assignedDate (optional)"
    ),
    due_date: str | None = Query(
        None, description="Filter by dueDate (optional)"),
):
    print(f"=== GET WEDDING TASKS DEBUG ===")
    print(f"Bearer token: {bearer_token}")
    print(f"Wedding session ID: {wedding_session_id}")
    print(
        f"Filters - Assignee: {assignee}, Status: {status}, Assigned date: {assigned_date}, Due date: {due_date}"
    )

    # 1) Authenticate + authorize - check if user is part of this wedding
    # session
    _auth_session(bearer_token, wedding_session_id)

    # 2) Base query - ALWAYS include wedding_session_id to get ALL tasks for
    # this session
    criteria: dict[str, Any] = {"wedding_session_id": wedding_session_id}
    print(f"Base criteria (all tasks in session): {criteria}")

    # 3) Apply optional filters ONLY if provided
    if assignee:
        criteria["assignee_id"] = assignee  # Filter by assignee's bearer token
        print(f"Added assignee filter: {assignee}")

    if status:
        criteria["taskStatus"] = status
        print(f"Added status filter: {status}")

    if assigned_date:
        criteria["assignedDate"] = assigned_date
        print(f"Added assigned_date filter: {assigned_date}")

    if due_date:
        criteria["dueDate"] = due_date
        print(f"Added due_date filter: {due_date}")

    # If no date filters provided, default behavior:
    # - Include ONLY today's tasks for items that have a taskType
    # - Include ALL tasks that do not have taskType, regardless of date
    if not assigned_date and not due_date:
        today_str = datetime.utcnow().date().isoformat()
        date_match_any = [
            {"dueDate": today_str},
            {"assignedDate": today_str},
            {"assignedDate": {"$regex": f"^{today_str}"}},
        ]
        criteria["$or"] = [
            {
                "$and": [
                    {"taskType": {"$exists": True}},
                    {"$or": date_match_any},
                ]
            },
            {"taskType": {"$exists": False}},
        ]
        print(
            f"Defaulting to today's taskType tasks and all no-taskType tasks. Today={today_str}"
        )

    print(f"Final query criteria: {criteria}")

    # 4) Fetch all matching tasks
    cursor = db["wedding_tasks"].find(criteria)
    tasks = list(cursor)
    filtered_tasks = []
    print(f"Found {len(tasks)} tasks")
    for i, task in enumerate(tasks):
        # logic to show tasks assigned to only bride made ( shoeing only
        # personal tasks of bride made)
        if "taskType" not in task:
            filtered_tasks.append(task)
        elif str(task.get("assignee_id")) == str(bearer_token):
            filtered_tasks.append(task)
        assignee_info = task.get("assignee_id", "Unassigned")
        print(
            f"Task {
                i +
                1}: {
                task.get('taskName')} - Assignee: {assignee_info} - Status: {
                task.get('taskStatus')}"
        )

    # 5) Clean and return
    cleaned_tasks = _clean_bson(filtered_tasks)
    print(f"Returning {len(cleaned_tasks)} tasks")

    return cleaned_tasks


def normalize_id(value):
    """Convert ObjectId or string to comparable string."""
    if isinstance(value, ObjectId):
        return str(value)
    return value


@app.get("/get-feeds", tags=["Feeds"])
async def get_feeds(
    bearer_token: str = Header(..., alias="bearerToken"),
    wedding_session_id: str = Header(..., alias="weddingSessionId"),
    post_id: Optional[str] = Header(None, alias="postId"),
):
    """Fetch feed posts: specific post if post_id provided, otherwise all posts for the session.

    Broader query: match by session_id (and post_type) only, not by bride bearer token.
    """
    _auth_session(bearer_token, wedding_session_id)

    # Verify requester exists (kept for auth/logging parity)
    user_doc = db["userOperations"].find_one({"bearer_token": bearer_token})
    print("User document:", user_doc)
    if not user_doc or "_id" not in user_doc:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    requester_user_id = normalize_id(user_doc["_id"])
    print("Requester User ID:", requester_user_id)

    # Helper to handle ObjectId/string conversion
    def handle_id_field(id_value, field_name):
        """Handle ID that could be ObjectId or string"""
        if not id_value:
            return None
        try:
            if len(str(id_value)) == 24:
                return ObjectId(str(id_value))
            else:
                return str(id_value)
        except (InvalidId, TypeError, ValueError):
            # If ObjectId conversion fails, keep as string
            return str(id_value)

    # Prepare base queries matching session only (and gallery post type)
    processed_session_id = handle_id_field(wedding_session_id, "session_id")

    # Base query with BRIDE'S bearer_token and session_id - try both formats
    base_queries = []

    # Try with session_id as processed value using BRIDE'S bearer token
    base_queries.append({"session_id": processed_session_id})

    # If processed_session_id is ObjectId, also try as string
    if isinstance(processed_session_id, ObjectId):
        base_queries.append({"session_id": str(processed_session_id)})
    elif isinstance(processed_session_id, str) and len(processed_session_id) == 24:
        try:
            base_queries.append({"session_id": ObjectId(processed_session_id)})
        except InvalidId:
            pass  # Skip if not a valid ObjectId

    # If post_id is provided, return specific document
    if post_id:
        processed_post_id = handle_id_field(post_id, "post_id")
        print(f"Looking for specific post: {processed_post_id}")

        record = None
        # Try different combinations of session_id and post_id formats
        for base_query in base_queries:
            # Try finding by _id first (assuming post_id refers to document
            # _id)
            post_queries = []

            # Try with post_id as _id
            if isinstance(processed_post_id, ObjectId):
                post_queries.append({**base_query, "_id": processed_post_id})
            elif isinstance(processed_post_id, str) and len(processed_post_id) == 24:
                try:
                    post_queries.append(
                        {**base_query, "_id": ObjectId(processed_post_id)}
                    )
                except InvalidId:
                    pass

            # Try with post_id as string _id (if document stores _id as string)
            post_queries.append({**base_query, "_id": str(processed_post_id)})

            # Also try if there's a separate post_id field in the document
            post_queries.append({**base_query, "post_id": processed_post_id})
            if isinstance(processed_post_id, ObjectId):
                post_queries.append(
                    {**base_query, "post_id": str(processed_post_id)})
            elif isinstance(processed_post_id, str) and len(processed_post_id) == 24:
                try:
                    post_queries.append(
                        {**base_query, "post_id": ObjectId(processed_post_id)}
                    )
                except InvalidId:
                    pass

            # Try each query until we find a record
            for query in post_queries:
                print(f"Trying query: {query}")
                record = db["voting_collection"].find_one(query)
                if record:
                    break

            if record:
                break

        print("Specific record:", record)
        if record:
            print("Specific record found:", record)
            return _clean_bson(record)
        else:
            raise HTTPException(
                status_code=404, detail=f"Post with ID {post_id} not found"
            )

    # Otherwise, return all documents for this session (any uploader)
    print("Fetching all records for wedding session (session_id + post_type only)")
    records = []
    for base_query in base_queries:
        print(f"Trying base query: {base_query}")
        records = list(db["voting_collection"].find(base_query))
        if records:
            break

    print(f"Found {len(records)} records")
    if records:
        cleaned_records = [_clean_bson(record) for record in records]
        return {"count": len(cleaned_records), "posts": cleaned_records}
    else:
        return {
            "count": 0,
            "posts": [],
            "message": "No records found for this Wedding Session",
        }


@app.post("/Like-Post", tags=["Feeds"])
async def save_like(
    session_id: str = Query(..., description="User session ID"),
    bearer_token: str = Query(...,
                              description="Bearer token for authentication"),
    post_id: str = Query(..., description="ID of the Post being voted on"),
):
    """
    Save the like record based on session_id, bearer_token, and post_id.
    """
    _auth_session(bearer_token, session_id)  # Fixed the function call

    # Helper function to handle ObjectId/string conversion
    def handle_id_field(id_value):
        """Handle ID that could be ObjectId or string"""
        if not id_value:
            return None

        try:
            # First try to create ObjectId if it looks like one
            if len(str(id_value)) == 24:  # Standard ObjectId length
                return ObjectId(str(id_value))
            else:
                return str(id_value)  # Keep as string
        except (InvalidId, TypeError, ValueError):
            # If ObjectId conversion fails, keep as string
            return str(id_value)

    # Fetch the user document by bearer_token
    user_doc = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user_doc or "_id" not in user_doc:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = user_doc["_id"]
    user_name = user_doc.get("name", "Unknown User")

    # Handle session_id and post_id formats
    processed_session_id = handle_id_field(session_id)
    processed_post_id = handle_id_field(post_id)

    # Build multiple query combinations to find the post
    post_queries = []

    # Different combinations of session_id and post_id formats
    session_variations = [processed_session_id]
    post_variations = [processed_post_id]

    # Add string versions if ObjectId
    if isinstance(processed_session_id, ObjectId):
        session_variations.append(str(processed_session_id))
    elif isinstance(processed_session_id, str) and len(processed_session_id) == 24:
        try:
            session_variations.append(ObjectId(processed_session_id))
        except InvalidId:
            pass

    if isinstance(processed_post_id, ObjectId):
        post_variations.append(str(processed_post_id))
    elif isinstance(processed_post_id, str) and len(processed_post_id) == 24:
        try:
            post_variations.append(ObjectId(processed_post_id))
        except InvalidId:
            pass

    # Create all possible query combinations
    for sess_id in session_variations:
        for p_id in post_variations:
            # Try with session_id field
            post_queries.append({"session_id": sess_id, "_id": p_id})
            post_queries.append({"session_id": sess_id, "post_id": p_id})

    # Try to find the post document
    post_doc = None
    used_query = None
    for query in post_queries:
        print(f"Trying query: {query}")
        post_doc = db["voting_collection"].find_one(query)
        if post_doc:
            used_query = query
            break

    print("Post document:", post_doc)
    if not post_doc:
        raise HTTPException(status_code=404,
                            detail="Post not found for this session.")

    # Initialize likes structure if it doesn't exist
    if "likes" not in post_doc:
        post_doc["likes"] = {"count": 0, "users": []}

    # Check if user has already liked this post
    user_already_liked = False
    for user in post_doc["likes"]["users"]:
        if user.get("bearer_token") == bearer_token:
            user_already_liked = True
            break

    if user_already_liked:
        return {"detail": "User has already liked this post."}

    # Add the like
    print(f"Likes before: {post_doc['likes']['count']}")
    new_like_count = post_doc["likes"]["count"] + 1
    new_user_entry = {
        "bearer_token": bearer_token,
        "name": user_name,
        "liked_at": datetime.now(timezone.utc),
    }

    # Create updated users list
    updated_users = post_doc["likes"]["users"].copy()
    updated_users.append(new_user_entry)

    print(f"Likes after: {new_like_count}")
    print(f"Users after: {updated_users}")

    # Update the document in the database using the same query that found it
    voting_collection = db["voting_collection"]

    # Use the same query structure that successfully found the document
    update_result = voting_collection.update_one(
        used_query,  # Use the exact query that found the document
        {
            "$set": {"likes.count": new_like_count, "likes.users": updated_users},
            "$currentDate": {"updated_at": True},
        },
    )

    # updating leader board
    leaderboard_coll = db["wedding_leaderboards"]

    # Find the document with the given session_id
    doc = leaderboard_coll.find_one({"session_id": session_id})

    if doc:
        # Case: Document exists
        leaderboard = doc.get("leaderboard", {})

        # Update or insert the bearer_token
        leaderboard[user_name] = leaderboard.get(user_name, 0) + 1

        # Push back update to DB
        leaderboard_coll.update_one(
            {"session_id": session_id}, {"$set": {"leaderboard": leaderboard}}
        )

    else:
        # Case: No document found, create new
        new_doc = {"session_id": session_id, "leaderboard": {user_name: 1}}
        leaderboard_coll.insert_one(new_doc)

    print(
        f"On Like , Leaderboard updated for session {session_id}, user {user_name}.")

    if update_result.modified_count == 0:
        raise HTTPException(status_code=500,
                            detail="Failed to update the like count")
    if update_result.modified_count > 0:
        try:
            # Get post details for notification
            post_category = post_doc.get("category", "a post")

            # Send notification to all users in wedding session
            fcm_sent_count = await send_feed_interaction_notification(
                wedding_session_id=session_id,
                interaction_type="like",
                actor_name=user_name,
                post_details=f"{post_category} gallery",
                comment_text=None,
            )
            print(f"Sent like notifications to {fcm_sent_count} users")

        except Exception as notification_error:
            print(f"Failed to send like notification: {notification_error}")

    await send_reaction_notification_webhook(
        wedding_session_id=session_id,
        joiner_name=user_name,
        joiner_user_id=user_id,
        reaction_type="like",
    )

    return {
        "status": "success",
        "message": "Like added successfully",
        "likes_count": new_like_count,
    }


@app.post("/Comment-Post", tags=["Feeds"])
async def save_comment(
    session_id: str = Query(..., description="User session ID"),
    bearer_token: str = Query(...,
                              description="Bearer token for authentication"),
    post_id: str = Query(..., description="ID of the Post being voted on"),
    comment: str = Body(..., description="Comment text"),
    parent_comment_id: Optional[str] = Query(
        None, description="ID of parent comment for replies"
    ),
):
    """
    Save the comment record with support for nested replies.
    """
    _auth_session(bearer_token, session_id)

    # Helper function to handle ObjectId/string conversion
    def handle_id_field(id_value):
        if not id_value:
            return None
        try:
            if len(str(id_value)) == 24:
                return ObjectId(str(id_value))
            else:
                return str(id_value)
        except (InvalidId, TypeError, ValueError):
            return str(id_value)

    # Helper function to find comment by ID in nested structure
    def find_comment_by_id(comments_data, comment_id):
        """Recursively find a comment by ID and return its path"""
        for i, comment in enumerate(comments_data):
            if comment.get("comment_id") == comment_id:
                return comment, [i]

            # Search in replies
            if "replies" in comment:
                found_comment, path = find_comment_by_id(
                    comment["replies"], comment_id)
                if found_comment:
                    return found_comment, [i, "replies"] + path

        return None, []

    # Helper function to add reply to nested structure
    def add_reply_to_comment(comments_data, parent_id, new_comment):
        """Add a reply to the specified parent comment"""
        parent_comment, path = find_comment_by_id(comments_data, parent_id)

        if not parent_comment:
            return False, "Parent comment not found"

        # Navigate to the parent comment using the path
        current = comments_data
        for step in path[:-1]:
            current = current[step]

        # Add reply to parent's replies array
        if "replies" not in current[path[-1]]:
            current[path[-1]]["replies"] = []

        current[path[-1]]["replies"].append(new_comment)
        return True, "Reply added successfully"

    # Fetch the user document by bearer_token
    user_doc = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user_doc or "_id" not in user_doc:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_name = user_doc.get("name", "Unknown User")
    user_id = str(user_doc["_id"])

    # Validate comment input
    if not comment or not comment.strip():
        raise HTTPException(status_code=400, detail="Comment cannot be empty")

    # Handle session_id and post_id formats
    processed_session_id = handle_id_field(session_id)
    processed_post_id = handle_id_field(post_id)

    # Build query combinations to find the post
    post_queries = []
    session_variations = [processed_session_id]
    post_variations = [processed_post_id]

    # Add alternative formats
    if isinstance(processed_session_id, ObjectId):
        session_variations.append(str(processed_session_id))
    elif isinstance(processed_session_id, str) and len(processed_session_id) == 24:
        try:
            session_variations.append(ObjectId(processed_session_id))
        except InvalidId:
            pass

    if isinstance(processed_post_id, ObjectId):
        post_variations.append(str(processed_post_id))
    elif isinstance(processed_post_id, str) and len(processed_post_id) == 24:
        try:
            post_variations.append(ObjectId(processed_post_id))
        except InvalidId:
            pass

    # Create query combinations
    for sess_id in session_variations:
        for p_id in post_variations:
            post_queries.append({"session_id": sess_id, "_id": p_id})
            post_queries.append({"session_id": sess_id, "post_id": p_id})
            post_queries.append(
                {"bearer_token": bearer_token, "session_id": sess_id, "_id": p_id}
            )
            post_queries.append(
                {"bearer_token": bearer_token, "session_id": sess_id, "post_id": p_id}
            )

    # Find the post document
    post_doc = None
    used_query = None
    for query in post_queries:
        post_doc = db["voting_collection"].find_one(query)
        if post_doc:
            used_query = query
            break

    if not post_doc:
        raise HTTPException(status_code=404,
                            detail="Post not found for this session.")

    # Initialize comments structure if it doesn't exist
    if "comments" not in post_doc:
        post_doc["comments"] = {"count": 0, "data": []}

    # Generate unique comment ID
    comment_id = str(uuid.uuid4())

    # Create new comment object
    new_comment = {
        "comment_id": comment_id,
        "bearer_token": bearer_token,
        "name": user_name,
        "comment": comment.strip(),
        "commented_at": datetime.now(timezone.utc),
        "parent_comment_id": parent_comment_id,
        "replies": [],
    }

    # Get current comments data
    current_comments = post_doc["comments"]["data"].copy()

    if parent_comment_id:
        # This is a reply to an existing comment
        success, message = add_reply_to_comment(
            current_comments, parent_comment_id, new_comment
        )
        if not success:
            raise HTTPException(status_code=400, detail=message)
    else:
        # This is a top-level comment
        current_comments.append(new_comment)

    # Update the document in the database
    update_result = db["voting_collection"].update_one(
        used_query,
        {
            "$inc": {"comments.count": 1},
            "$set": {"comments.data": current_comments},
            "$currentDate": {"updated_at": True},
        },
    )

    if update_result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to add comment")

    # Get updated count
    updated_doc = db["voting_collection"].find_one(
        used_query, {"comments.count": 1})
    new_count = updated_doc["comments"]["count"] if updated_doc else 1

    if not parent_comment_id:  # we need to update the score for only parent comments
        # updating leader board
        leaderboard_coll = db["wedding_leaderboards"]

        # Find the document with the given session_id
        doc = leaderboard_coll.find_one({"session_id": session_id})

        if doc:
            # Case: Document exists
            leaderboard = doc.get("leaderboard", {})

            # Update or insert the bearer_token
            leaderboard[user_name] = leaderboard.get(user_name, 0) + 2

            # Push back update to DB
            leaderboard_coll.update_one(
                {"session_id": session_id}, {
                    "$set": {"leaderboard": leaderboard}}
            )

        else:
            # Case: No document found, create new
            new_doc = {"session_id": session_id, "leaderboard": {user_name: 2}}
            leaderboard_coll.insert_one(new_doc)

        print(
            f"On comment , Leaderboard updated for session {session_id}, user {user_name}."
        )

    if update_result.modified_count > 0:
        try:
            # Get post details for notification
            post_category = post_doc.get("category", "a post")

            # Send notification to all users in wedding session
            fcm_sent_count = await send_feed_interaction_notification(
                wedding_session_id=session_id,
                interaction_type="comment",
                actor_name=user_name,
                post_details=f"{post_category} gallery",
                comment_text=comment.strip(),
            )
            print(f"Sent comment notifications to {fcm_sent_count} users")

        except Exception as notification_error:
            print(f"Failed to send comment notification: {notification_error}")

    await send_reaction_notification_webhook(
        wedding_session_id=session_id,
        joiner_name=user_name,
        joiner_user_id=user_id,
        reaction_type="comment",
    )

    return {
        "status": "success",
        "message": "Comment added successfully",
        "comment_id": comment_id,
        "comments_count": new_count,
        "is_reply": parent_comment_id is not None,
        "parent_comment_id": parent_comment_id,
    }


async def get_image_details_by_uuid(image_uuid: str):
    """
    Fetch the image details from the database based on the provided image UUID.
    Returns a dictionary with image details (description, category, URL) if found, or None if not.
    """
    image_doc = db["user_image_urls"].find_one({"images.image_id": image_uuid})

    if image_doc:
        # Find the image inside the images array
        for img in image_doc["images"]:
            if img["image_id"] == image_uuid:
                return {
                    "description": img.get("description", "No description available"),
                    "url": img.get("url", "URL not available"),
                    "category": img.get("category", "General"),
                }

    # If image is not found, return None
    return None


@app.post("/process-task", tags=["Leaderboard"])
async def process_task(task_id: str = Body(..., embed=True)):
    """
    • Look up task by its UUID `task_id`
    • If still pending → mark done & +5 points to that assignee
      in the per-wedding embedded leaderboard map
    """
    # ── 1. fetch the task by public UUID ───────────────────────────
    task = db["wedding_tasks"].find_one({"task_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    status = task.get("taskStatus", "unknown").lower()
    assignee_bearer_token = task.get(
        "assignee_id"
    )  # This is the assignee's bearer token
    bearer_token = task.get("bearer_token")  # This is the bride's bearer token
    wedding_session_id = task.get("wedding_session_id")

    print(f"=== PROCESS TASK DEBUG ===")
    print(f"Task ID: {task_id}")
    print(f"Status: {status}")
    print(f"Assignee Bearer Token: {assignee_bearer_token}")
    print(f"Bride Bearer Token: {bearer_token}")
    print(f"Wedding Session ID: {wedding_session_id}")

    # ── 2. Validate required fields ────────────────────────────────
    if not bearer_token:
        raise HTTPException(
            status_code=400,
            detail="Task missing bearer_token")

    if not wedding_session_id:
        raise HTTPException(status_code=400,
                            detail="Task missing wedding_session_id")

    if not assignee_bearer_token:
        raise HTTPException(status_code=400,
                            detail="Task missing assignee information")

    # ── 3. only award points once (pending → done) ────────────────
    if status == "pending":
        print(f"Processing pending task...")

        # mark task done
        db["wedding_tasks"].update_one(
            {"task_id": task_id},
            {"$set": {"taskStatus": "done", "completedAt": datetime.utcnow()}},
        )
        print(f"✓ Task marked as done")

        # +5 to leaderboard using assignee's bearer token as the key
        db["wedding_leaderboards"].update_one(
            {
                "bearer_token": bearer_token,  # Bride's bearer token
                "session_id": wedding_session_id,  # Wedding session ID
            },
            {
                "$inc": {
                    f"leaderboard.{assignee_bearer_token}": 5
                },  # Assignee's bearer token as key
                "$setOnInsert": {
                    "bearer_token": bearer_token,
                    "session_id": wedding_session_id,
                    "created_at": datetime.utcnow(),
                },
            },
            upsert=True,
        )
        print(f"✓ Added 5 points to assignee {assignee_bearer_token}")

        return {
            "task_id": str(task_id),
            "points_awarded": 5,
            "new_status": "done",
            "assignee_bearer_token": str(assignee_bearer_token),
        }

    # already processed or not pending
    print(
        f"Task already processed or not pending. Current status: {
            task.get(
                'taskStatus',
                'unknown')}"
    )
    return {
        "task_id": str(task_id),
        "status": task.get("taskStatus", "unknown"),
        "assignee_bearer_token": (
            str(assignee_bearer_token) if assignee_bearer_token else None
        ),
    }


@app.get("/get-leaderboard", tags=["Leaderboard"])
async def get_leaderboard(
    bearer_token: str = Query(...,
                              description="Bearer token (bride or assignee)"),
    wedding_session_id: str = Query(..., description="Wedding session ID"),
) -> List[Dict[str, Union[str, int]]]:
    """
    Retrieves the leaderboard sorted by descending points for the given wedding session.
    Works for both bride and assignees (participants).
    """
    bearer_token = str(bearer_token.strip())
    if not bearer_token:
        logger.warning("Invalid bearer token format received")
        raise HTTPException(
            status_code=400,
            detail="Invalid bearer token format")

    token = bearer_token.replace("Bearer ", "")

    # Use _auth_session to validate access (works for both bride and
    # participants)
    _auth_session(token, wedding_session_id)

    # Now we need to find the bride's bearer token to query the leaderboard
    # (since leaderboards are stored under the bride's bearer token)
    wedding_session = db["weddingSessionOperations"].find_one(
        {"_id": ObjectId(wedding_session_id)}
    )
    if not wedding_session:
        raise HTTPException(
            status_code=404,
            detail="Wedding session not found")

    # Get the bride's user ID
    bride_user_id = wedding_session["users"].get("bride")
    if not bride_user_id:
        raise HTTPException(
            status_code=404,
            detail="Bride not found in session")

    # Get the bride's bearer token
    bride_doc = db["userOperations"].find_one({"_id": bride_user_id})
    if not bride_doc:
        raise HTTPException(
            status_code=404,
            detail="Bride user record not found")

    bride_bearer_token = bride_doc.get("bearer_token")
    if not bride_bearer_token:
        raise HTTPException(
            status_code=404,
            detail="Bride bearer token not found")

    print(f"=== LEADERBOARD DEBUG ===")
    print(f"Requester token: {token}")
    print(f"Bride token: {bride_bearer_token}")
    print(f"Wedding session ID: {wedding_session_id}")

    collection = db["wedding_leaderboards"]

    try:
        # Query using bride's bearer token and session ID
        doc = collection.find_one({"session_id": wedding_session_id})
    except Exception as e:
        logger.exception("Failed to query leaderboard collection")
        raise HTTPException(status_code=500, detail="Database error")

    if not doc or "leaderboard" not in doc:
        logger.info(
            f"Leaderboard not found for bride token: {bride_bearer_token} and session: {wedding_session_id}"
        )
        raise HTTPException(status_code=404, detail="Leaderboard not found")

    leaderboard = doc["leaderboard"]
    logger.debug(f"Raw leaderboard: {leaderboard}")

    sorted_leaderboard = sorted(
        leaderboard.items(), key=lambda item: item[1], reverse=True
    )

    logger.info(
        f"Sorted leaderboard generated for session: {wedding_session_id}")

    # Fetch user names for each assignee bearer token
    result = []
    for participant_name, points in sorted_leaderboard:
        result.append({"name": participant_name, "points": points})

    return result


@app.post("/update_likes_and_comments", tags=["Reactions"])
async def update_likes_and_comments(
    bearer_token: str = Header(...),
    session_id: Optional[str] = Header(None),
    title: str = Header(...),
    reaction_type: str = Header(...),  # "like" or "comment"
    assignee_id: str = Header(...),
    comment_text: Optional[str] = Header(None),
):
    """
    Updates the like or comment for a specific collection item.
    """

    if reaction_type == "comment" and not comment_text:
        raise HTTPException(
            status_code=400,
            detail="comment_text is required when reaction_type is 'comment'",
        )

    collection = db["voting_collection"]

    # Find the matching document
    document = collection.find_one(
        {"bearer_token": bearer_token, "session_id": session_id}
    )

    if not document:
        return {"status": "error", "message": "Document not found."}

    recommendations = document.get("recommendations", {})
    updated = False

    # Update the matched recommendation
    for category, entries in recommendations.items():
        for rec in entries:
            if rec.get("title") == title:
                if reaction_type == "like":
                    if "likes" not in rec:
                        rec["likes"] = {}
                    rec["likes"][assignee_id] = "liked"

                elif reaction_type == "comment":
                    if "comments" not in rec:
                        rec["comments"] = {}
                    rec["comments"][assignee_id] = comment_text
                updated = True
                # Pass session_id to update_leaderboard
                update_response = update_leaderboard(
                    bearer_token, session_id, assignee_id, reaction_type
                )
                break
        if updated:
            break

    if not updated:
        return {
            "status": "error",
            "message": f"No recommendation found with title '{title}'.",
        }

    # Save the updated document
    collection.update_one(
        {"_id": document["_id"]}, {
            "$set": {"recommendations": recommendations}}
    )

    return {
        "status": "success",
        "message": f"{reaction_type.capitalize()} updated successfully.",
        "title": title,
        "assignee_id": assignee_id,
        "update_response": update_response,
    }


def update_leaderboard(
    bearer_token: str, session_id: str, assignee_id: str, action_type: str
):
    db = get_db()

    leaderboard_collection = db["wedding_leaderboards"]

    # Define points based on action
    points_map = {"like": 2, "comment": 3}

    if action_type not in points_map:
        return {
            "status": "error",
            "message": "Invalid action_type. Must be 'like' or 'comment'.",
        }

    points_to_add = points_map[action_type]

    # Try to find the document for the given bearer_token AND session_id
    user_doc = leaderboard_collection.find_one(
        {"bearer_token": bearer_token, "session_id": session_id}
    )

    if not user_doc:
        # Create new document if user doesn't exist for this session
        new_doc = {
            "session_id": session_id,
            "bearer_token": bearer_token,
            "leaderboard": {assignee_id: points_to_add},
            "created_at": datetime.utcnow(),
        }
        leaderboard_collection.insert_one(new_doc)
        return {
            "status": "success",
            "message": f"Created new leaderboard entry for {assignee_id} with {points_to_add} points.",
        }

    # User document exists for this session, update leaderboard field
    leaderboard = user_doc.get("leaderboard", {})

    # Update points for the assignee
    current_points = leaderboard.get(assignee_id, 0)
    leaderboard[assignee_id] = current_points + points_to_add

    # Update the document
    leaderboard_collection.update_one(
        {"bearer_token": bearer_token, "session_id": session_id},
        {"$set": {"leaderboard": leaderboard}},
    )

    return {
        "status": "success",
        "message": f"Updated {assignee_id} with {points_to_add} points.",
        "total_points": leaderboard[assignee_id],
    }


def _convert_objectid_to_str(data):
    """Recursively convert all ObjectIds in a dictionary to strings."""
    if isinstance(data, dict):
        return {key: _convert_objectid_to_str(
            value) for key, value in data.items()}
    elif isinstance(data, list):
        return [_convert_objectid_to_str(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data


# ================================================================================================================
# def _auth_and_get_ids(bearer_token: str, wedding_session_id: str):
#     user = db["userOperations"].find_one({"bearer_token": bearer_token})
#     if not user:
#         raise HTTPException(401, "Invalid bearer token")
#     user_id = user["_id"]

#     try:
#         sess_oid = ObjectId(wedding_session_id)
#     except Exception:
#         raise HTTPException(400, "Malformed wedding_session_id")

#     session = db["weddingSessionOperations"].find_one(
#         {"_id": sess_oid, "users.bride": user_id}
#     )
#     if not session:
#         raise HTTPException(404, "Wedding session not found or not authorised")

#     return user_id, sess_oid


def auth_and_get_ids(
    bearer_token: str, wedding_session_id: Optional[str]
) -> Tuple[Any, Optional[ObjectId]]:
    # --- authenticate the user ---
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    user_id = user["_id"]

    # --- only validate session if provided ---
    if wedding_session_id is not None:
        try:
            sess_oid = ObjectId(wedding_session_id)
        except (bson_errors.InvalidId, TypeError):
            raise HTTPException(
                status_code=400,
                detail="Malformed wedding_session_id")

        # Check if user is either bride or in participants
        session = db["weddingSessionOperations"].find_one(
            {
                "_id": sess_oid,
                "$or": [
                    {"users.bride": user_id},  # User is the bride
                    {
                        f"users.participants.{key}": str(user_id) for key in ["$exists"]
                    },  # This won't work directly
                ],
            }
        )

        # Better approach - get the session first, then check authorization
        session = db["weddingSessionOperations"].find_one({"_id": sess_oid})
        if not session:
            raise HTTPException(
                status_code=404,
                detail="Wedding session not found")

        # Check if user is authorized (bride or participant)
        is_bride = session.get("users", {}).get("bride") == user_id

        # Check if user is in participants (participants store user_id as
        # string values)
        participants = session.get("users", {}).get("participants", {})
        is_participant = str(user_id) in participants.values()

        if not (is_bride or is_participant):
            raise HTTPException(
                status_code=403, detail="Not authorized to access this wedding session"
            )
    else:
        sess_oid = None

    return user_id, sess_oid


# @app.post("/upload-image", tags=["Wedding Operations"])
# async def upload_image(
#     file: UploadFile = File(..., description="JPEG / PNG photo"),
#     description: str = Header(..., description="Short caption"),
#     bearer_token: str = Header(...),
#     wedding_session_id: Optional[str] = Header(None, description="(Optional) existing wedding-session ID"),
#     category: Optional[str] = Header(None, description="Optional image category")  # New category field
# ):
#     # --- Validate bearer_token first
#     if not bearer_token:
#         return JSONResponse(
#             status_code=401,
#             content={"status": "error", "message": "Missing bearer_token header."},
#         )

#     # Lookup user for bearer_token
#     user = db["userOperations"].find_one({"bearer_token": bearer_token})
#     if not user or "_id" not in user:
#         return JSONResponse(
#             status_code=401,
#             content={"status": "error", "message": "Invalid bearer token."},
#         )

#     user_id = str(user["_id"])
#     sess_oid = None

#     # --- Only validate session if wedding_session_id is provided
#     if wedding_session_id:
#         try:
#             _auth_session(bearer_token=bearer_token, wedding_session_id=wedding_session_id)
#             sess_oid = ObjectId(wedding_session_id)
#         except Exception:
#             return JSONResponse(
#                 status_code=400,
#                 content={"status": "error", "message": "Not authorized to access this wedding session"},
#             )

#     # 1) Upload to S3 --------------------------------------------------
#     try:
#         ext = file.filename.split(".")[-1]
#         image_id = str(uuid4()).replace('-', '_') + '_imgid'  # Unique ID for the image
#         key = f"{image_id}.{ext}"

#         s3_client.upload_fileobj(
#             file.file,
#             S3_BUCKET_NAME,
#             key,
#             ExtraArgs={"ContentType": file.content_type},
#         )
#     except NoCredentialsError:
#         raise HTTPException(500, "AWS credential error")
#     except Exception as e:
#         raise HTTPException(500, f"Upload failed: {e}")

#     file_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"

#     # 2) Build MongoDB query based on whether session_id is provided
#     filter_query = {"bearer_token": bearer_token}

#     # Fields for MongoDB update
#     set_on_insert = {
#         "bearer_token": bearer_token,
#         "user_id": user_id,
#         "created_at": now_ist,
#     }

#     # Only add session-related fields if wedding_session_id is provided
#     if wedding_session_id and sess_oid:
#         filter_query["wedding_session_id"] = sess_oid
#         set_on_insert["wedding_session_id"] = sess_oid
#     else:
#         # For uploads without session, ensure we're querying documents without session_id
#         filter_query["$or"] = [
#             {"wedding_session_id": {"$exists": False}},
#             {"wedding_session_id": None},
#             {"wedding_session_id": ""}
#         ]

#     # Add the category if provided
#     if category:
#         set_on_insert["category"] = category

#     db["user_image_urls"].update_one(
#         filter_query,
#         {
#             "$setOnInsert": set_on_insert,
#             "$push": {
#                 "images": {
#                     "description": description,
#                     "url": file_url,
#                     "uploaded_at": now_ist,
#                     "image_id": image_id,  # Save the image ID here as well
#                     "category": category,  # Save the category if provided
#                 }
#             },
#             "$set": {"updated_at": now_ist},
#         },
#         upsert=True,
#     )

#     return {
#         "message": "Image uploaded successfully",
#         "image_url": file_url,
#         "description": description,
#         "session_id": wedding_session_id,
#         "image_id": image_id,  # Return the unique image ID
#         "category": category,  # Return the category (if provided)
#     }


# Function to upload image to GCP
def upload_image_to_gcp(file_obj, bucket_name: str, new_file_name: str):
    private_key = os.getenv("PRIVATE_KEY")
    client_email = os.getenv("CLIENT_EMAIL")
    project_id = os.getenv("PROJECT_ID")

    if not private_key or not client_email or not project_id:
        raise ValueError("Missing credentials. Please check your .env file.")

    # Create credentials object using the private key and email
    credentials = service_account.Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": os.getenv("PRIVATE_KEY_ID"),
            "private_key": private_key,
            "client_email": client_email,
            "client_id": os.getenv("CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
        }
    )

    # Create a storage client with the credentials
    storage_client = storage.Client(
        credentials=credentials,
        project=project_id)

    # Get the GCP bucket
    bucket = storage_client.bucket(bucket_name)

    # Define the custom name for the file and specify the folder (e.g.,
    # test/new-lord-shiva.jpg)
    new_blob_name = (
        f"test/{new_file_name}"  # Specify folder (test/) and custom file name
    )

    # Define the blob (file object in the bucket) with the new name
    blob = bucket.blob(new_blob_name)

    # Upload the file to GCP bucket directly from the file object
    blob.upload_from_file(
        file_obj, content_type="image/jpeg"
    )  # Adjust content_type if necessary

    # The file will be publicly accessible automatically due to the
    # bucket-level permission

    # Get the public URL of the uploaded image
    public_url = blob.public_url
    return public_url


# Helper function to extract category from various description formats
def extract_category_from_description(description: str) -> Optional[str]:
    """
    Extract category from description using various patterns:
    1. JSON format: {"category": "Bridal_party", "image": "image00_1.jpg"}
    2. Escaped JSON: "{\"category\": \"Bridal_party\", \"image\": \"image00_1.jpg\"}"
    3. Simple format: category: Bridal_party
    4. Key-value format: category=Bridal_party
    """
    if not description:
        return None

    print(f"Extracting category from description: {description}")

    # First, try to unescape the string and parse as JSON
    try:
        # Handle escaped quotes
        unescaped = description.replace('\\"', '"').replace("\\'", "'")
        print(f"Unescaped description: {unescaped}")

        # Try to parse as JSON
        if unescaped.strip().startswith(
                "{") and unescaped.strip().endswith("}"):
            try:
                import json

                parsed = json.loads(unescaped)
                if isinstance(parsed, dict) and "category" in parsed:
                    category = parsed["category"]
                    print(f"Found category from JSON: {category}")
                    return category.strip()
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
    except Exception as e:
        print(f"Error in JSON extraction: {e}")

    # Fallback to regex patterns
    patterns = [
        # JSON format with quotes
        r'\{\s*["\']?category["\']?\s*:\s*["\']([^"\']+)["\']',
        # Escaped JSON format
        r'\{\\\s*\\"category\\"\s*:\s*\\"([^"\\]+)\\"',
        # Simple colon format
        r"category\s*:\s*([^,\s}]+)",
        # Equals format
        r"category\s*=\s*([^,\s}]+)",
        # Parentheses format
        r"\(\s*([^)]+)\s*\)",
    ]

    for i, pattern in enumerate(patterns):
        try:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                category = match.group(1).strip()
                print(f"Found category using pattern {i}: {category}")
                return category
        except Exception as e:
            print(f"Pattern {i} failed: {e}")
            continue

    print("No category found in description")
    return None


@app.post("/upload-image", tags=["Wedding Operations"])
async def upload_image(
    file: UploadFile = File(..., description="JPEG / PNG photo"),
    description: str = Header(..., description="Short caption"),
    bearer_token: str = Header(...),
    wedding_session_id: Optional[str] = Header(
        None, description="(Optional) existing wedding-session ID"
    ),
    category: Optional[str] = Header(
        None, description="Optional image category"
    ),  # New category field
    task_id: Optional[str] = Header(
        None,
        alias="task-id",
        description="Optional: task to auto-complete on upload (kebab-case)",
    ),
    task_id_alt: Optional[str] = Header(
        None,
        alias="task_id",
        description="Optional: task to auto-complete on upload (snake_case)",
    ),
):
    # --- Validate bearer_token first
    if not bearer_token:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": "Missing bearer_token header."},
        )

    # Lookup user for bearer_token
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user or "_id" not in user:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid bearer token."},
        )

    user_id = str(user["_id"])
    sess_oid = None

    # --- Only validate session if wedding_session_id is provided
    if wedding_session_id:
        try:
            _auth_session(
                bearer_token=bearer_token, wedding_session_id=wedding_session_id
            )
            sess_oid = ObjectId(wedding_session_id)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Not authorized to access this wedding session",
                },
            )

    # --- Extract category from description or use header category ---
    extracted_category = extract_category_from_description(description)
    final_category = extracted_category or category or "general"

    print(f"Final category determined: {final_category}")

    # 1) Upload to GCP (primary) with S3 fallback (if configured) ------------
    try:
        ext = file.filename.split(".")[-1] if file.filename else "jpg"
        image_id = str(uuid4()).replace("-", "_") + \
            "_imgid"  # Unique ID for the image
        key = f"{image_id}.{ext}"

        # Try GCP upload first
        try:
            # Reset file pointer for GCP upload
            file.file.seek(0)
            file_url = upload_image_to_gcp(
                file.file, "sayyes-images", f"{image_id}.{ext}"
            )
            print(f"Successfully uploaded to GCP: {file_url}")
        except Exception as gcp_error:
            print(f"GCP upload failed: {gcp_error}")

            # Only try S3 if credentials are properly configured
            try:
                # Check if S3 credentials are available
                if not (
                    os.getenv("AWS_ACCESS_KEY_ID")
                    and os.getenv("AWS_SECRET_ACCESS_KEY")
                ):
                    raise HTTPException(
                        500,
                        "No valid upload service available. GCP failed and S3 not configured.",
                    )

                print("Attempting S3 fallback...")
                # Reset file pointer for S3 upload
                file.file.seek(0)
                s3_client.upload_fileobj(
                    file.file,
                    S3_BUCKET_NAME,
                    key,
                    ExtraArgs={"ContentType": file.content_type},
                )
                file_url = (
                    f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{key}"
                )
                print(f"Successfully uploaded to S3: {file_url}")

            except Exception as s3_error:
                print(f"S3 upload also failed: {s3_error}")
                raise HTTPException(
                    500,
                    f"Both GCP and S3 upload failed. GCP error: {gcp_error}. S3 error: {s3_error}",
                )

    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {e}")

    # 2) Build MongoDB query for user_image_urls collection
    filter_query = {"bearer_token": bearer_token}

    # Fields for MongoDB update
    set_on_insert = {
        "bearer_token": bearer_token,
        "user_id": user_id,
        "created_at": now_ist,
    }

    # Only add session-related fields if wedding_session_id is provided
    if wedding_session_id and sess_oid:
        filter_query["wedding_session_id"] = sess_oid
        set_on_insert["wedding_session_id"] = sess_oid
    else:
        # For uploads without session, ensure we're querying documents without
        # session_id
        filter_query["$or"] = [
            {"wedding_session_id": {"$exists": False}},
            {"wedding_session_id": None},
            {"wedding_session_id": ""},
        ]

    # Add the category if provided
    if final_category:
        set_on_insert["category"] = final_category

    # --- Prepare the image doc (what we store in arrays)
    image_doc = {
        "description": description,  # keep the exact incoming string
        "url": file_url,
        "uploaded_at": now_ist,
        "image_id": image_id,
        "category": final_category or None,  # raw value as provided
    }

    # --- New storage structure: images_by_category.<category_key> array
    # Also keep legacy flat 'images' array for backward compatibility
    # (optional)
    db["user_image_urls"].update_one(
        filter_query,
        {
            "$setOnInsert": set_on_insert,
            "$push": {
                # Legacy (keep for backward compatibility); remove if you want
                # to fully migrate:
                "images": image_doc,
                # New structure: dictionary of arrays keyed by category
                f"images_by_category.{final_category}": image_doc,
            },
            "$set": {"updated_at": now_ist},
        },
        upsert=True,
    )

    # 3) Save to voting_collection if wedding_session_id is provided ---------
    voting_collection_updated = False
    if wedding_session_id:
        try:
            print(
                f"Starting voting collection save for session: {wedding_session_id}, category: {final_category}"
            )

            # Check if a voting document already exists for this session,
            # bearer token, AND specific category
            voting_query = {
                "bearer_token": bearer_token,
                "session_id": wedding_session_id,
                "category": final_category,  # Single category, not array
                "post_type": "gallery",
            }
            print(f"Voting query: {voting_query}")

            existing_voting_doc = db["voting_collection"].find_one(
                voting_query)
            print(
                f"Existing voting doc found for category '{final_category}': {
                    existing_voting_doc is not None}"
            )

            current_time = datetime.now(timezone.utc)

            if existing_voting_doc:
                print(
                    f"Updating existing voting document for category: {final_category}"
                )
                # Update existing document - add image to the images array
                update_query = {
                    "$push": {"images": file_url},
                    "$set": {"timestamp": current_time.isoformat()},
                }

                print(f"Update query: {update_query}")
                result = db["voting_collection"].update_one(
                    {"_id": existing_voting_doc["_id"]}, update_query
                )
                print(
                    f"Update result - matched: {
                        result.matched_count}, modified: {
                        result.modified_count}"
                )

                if result.modified_count > 0:
                    print(
                        f"Successfully updated existing voting document for category: {final_category}"
                    )
                else:
                    print("Warning: Update matched document but didn't modify anything")

            else:
                print(
                    f"Creating new voting document for category: {final_category}")
                # Create new voting document for this specific category in your
                # specified format
                new_voting_doc = {
                    "bearer_token": bearer_token,
                    "session_id": wedding_session_id,
                    "user_id": ObjectId(user_id),
                    "post_type": "gallery",
                    "category": final_category,  # Single category string
                    "images": [file_url],  # Array of image URLs
                    "likes": {"count": 0, "users": []},
                    "comments": {"count": 0, "data": []},
                    "timestamp": current_time.isoformat(),
                }

                print(f"New voting document: {new_voting_doc}")
                result = db["voting_collection"].insert_one(new_voting_doc)
                print(f"Insert result - inserted_id: {result.inserted_id}")
                print(
                    f"Successfully created new voting document for category: {final_category}"
                )

            voting_collection_updated = True
            print("Voting collection update completed successfully")

        except Exception as e:
            print(f"Error saving to voting_collection: {e}")
            import traceback

            print(f"Full traceback: {traceback.format_exc()}")
            # Don't fail the entire upload if voting collection save fails
            voting_collection_updated = False

    if wedding_session_id:
        try:
            # Get uploader's name for the notification
            uploader_name = user.get("name", "Someone")

            # Send notification to all participants
            fcm_sent_count = await send_image_upload_notification(
                wedding_session_id=wedding_session_id,
                uploader_name=uploader_name,
                image_description=description,
                category=final_category,
            )
            print(
                f"Sent FCM notifications to {fcm_sent_count} participants for image upload"
            )

        except Exception as notification_error:
            print(
                f"Failed to notify participants about image upload: {notification_error}"
            )

    if wedding_session_id:
        # sending in app notification after uploding image
        await send_reaction_notification_webhook(
            wedding_session_id=wedding_session_id,
            joiner_name=uploader_name,
            joiner_user_id=user_id,
            reaction_type="uploaded image",
        )

    # If a task identifier is provided, mark that task as completed
    effective_task_id = task_id or task_id_alt
    if effective_task_id:
        try:
            task_doc = db["wedding_tasks"].find_one(
                {"task_id": effective_task_id})
            if not task_doc:
                print(
                    f"[upload-image] No task found for task_id={effective_task_id}")
            else:
                # Optional: ensure same session if provided
                if wedding_session_id:
                    try:
                        task_sess = str(task_doc.get("wedding_session_id"))
                        if str(task_sess) != str(wedding_session_id):
                            print(
                                f"[upload-image] task session mismatch for task_id={effective_task_id}; skipping completion"
                            )
                            task_doc = None
                    except Exception:
                        pass
                if task_doc:
                    # Authorization: assignee or bride (creator) can complete
                    if bearer_token not in [
                        task_doc.get("assignee_id"),
                        task_doc.get("bearer_token"),
                    ]:
                        print(
                            f"[upload-image] bearer not authorized to complete task_id={effective_task_id}; skipping"
                        )
                    else:
                        db["wedding_tasks"].update_one(
                            {"_id": task_doc["_id"]},
                            {
                                "$set": {
                                    "taskStatus": "done",
                                    "completedAt": datetime.utcnow(),
                                    "completedBy": bearer_token,
                                }
                            },
                        )
                        print(
                            f"[upload-image] Marked task task_id={effective_task_id} as done"
                        )
        except Exception as e:
            print(
                f"[upload-image] Error completing task task_id={effective_task_id}: {e}"
            )

    return {
        "message": "Image uploaded successfully",
        "image_url": file_url,
        "description": description,
        "session_id": wedding_session_id,
        "image_id": image_id,  # Return the unique image ID
        "category": final_category,  # Return the determined category
        "voting_collection_updated": voting_collection_updated,
    }


@app.post("/venue-summaries", tags=["Wedding Recommendations"])
async def get_venue_summaries(
    titles: List[str] = Body(
        ..., description="List of venue titles to retrieve summaries for"
    ),
    bearer_token: str = Header(...,
                               description="Bearer token for authentication"),
    wedding_session_id: str = Header(
        ..., alias="wedding-session-id", description="Wedding session ID"
    ),
):

    # Authenticate user
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": "Invalid bearer token",
                "data": []},
        )

    user_id = str(user["_id"])

    # Validate wedding session access
    try:
        _auth_session(bearer_token, wedding_session_id)
    except BaseException:
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": "Not authorized to access this wedding session",
                "data": [],
            },
        )

    # Validate input
    if not titles or not isinstance(titles, list):
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": "Please provide a list of venue titles",
                "data": [],
            },
        )

    try:
        # Query the onboarding_recommendations collection
        collection = db["onboarding_recommendations"]

        # Find documents where title matches any of the provided titles
        found_summaries = list(
            collection.find(
                {"title": {"$in": titles}},
                {"_id": 0, "title": 1, "recommendation_summary": 1},
            )
        )

        # Create lookup dictionary
        found_dict = {
            item["title"]: item.get("recommendation_summary", "")
            for item in found_summaries
        }

        # Build response for ALL requested titles
        response_data = []
        for title in titles:
            response_data.append(
                {"title": title,
                    "recommendation_summary": found_dict.get(title, "")}
            )

        # ===== UPDATE THE ONBOARDING_RECOMMENDATIONS DOCUMENTS =====
        # Add user info to each venue document that was liked
        for title in titles:
            result = collection.update_one(
                {"title": title},
                {
                    "$addToSet": {
                        "liked_by_users": {
                            "user_id": user_id,
                            "bearer_token": bearer_token,
                            "wedding_session_id": wedding_session_id,
                            "liked_at": datetime.utcnow(),
                        }
                    }
                },
            )
            print(
                f"Updated venue '{title}': matched={
                    result.matched_count}, modified={
                    result.modified_count}"
            )
        # ============================================================

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_data,
                "requested_count": len(titles),
                "found_count": len(found_summaries),
            },
        )

    except Exception as e:
        print(f"Error fetching venue summaries: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "success": False,
                "error": "Internal server error while fetching venue summaries",
                "data": [],
            },
        )


async def get_user_liked_venues_from_documents(
    user_id: str, bearer_token: str, wedding_session_id: str
):
    """
    Get venues that this user has liked by checking the updated onboarding_recommendations documents
    """
    try:
        print(
            f"Checking onboarding_recommendations documents for user: {user_id}")

        collection = db["onboarding_recommendations"]

        # Find all documents where this user is in the liked_by_users array
        liked_venues = list(
            collection.find(
                {
                    "liked_by_users": {
                        "$elemMatch": {
                            "user_id": user_id,
                            "bearer_token": bearer_token,
                            "wedding_session_id": wedding_session_id,
                        }
                    }
                },
                {"_id": 0, "title": 1, "recommendation_summary": 1},
            )
        )

        print(f"Found {len(liked_venues)} venues liked by user {user_id}")

        return {"success": True, "data": liked_venues,
                "count": len(liked_venues)}

    except Exception as e:
        print(f"Error getting user liked venues from documents: {e}")
        return {"success": False, "error": str(e), "data": []}


async def get_user_liked_venues_with_summaries(
    user_id: str, bearer_token: str, wedding_session_id: str
):
    """
    Get venues that this user has previously liked with their summaries from onboarding_recommendations
    """
    try:
        print(f"Checking liked venues for user: {user_id}")

        # Get user's liked venue titles from user_venue_likes collection
        user_likes_collection = db["user_venue_likes"]
        user_likes_doc = user_likes_collection.find_one(
            {
                "user_id": user_id,
                "bearer_token": bearer_token,
                "wedding_session_id": wedding_session_id,
            }
        )

        if not user_likes_doc or not user_likes_doc.get("liked_venues"):
            print(f"No liked venues found for user {user_id}")
            return {"success": True, "data": [], "count": 0}

        liked_venue_titles = user_likes_doc["liked_venues"]
        print(
            f"Found {
                len(liked_venue_titles)} liked venue titles: {liked_venue_titles}"
        )

        # Fetch summaries from onboarding_recommendations collection
        recommendations_collection = db["onboarding_recommendations"]
        venue_summaries = list(
            recommendations_collection.find(
                {"title": {"$in": liked_venue_titles}},
                {"_id": 0, "title": 1, "recommendation_summary": 1},
            )
        )

        print(f"Found summaries for {len(venue_summaries)} venues")

        return {"success": True, "data": venue_summaries,
                "count": len(venue_summaries)}

    except Exception as e:
        print(f"Error getting user liked venues: {e}")
        return {"success": False, "error": str(e), "data": []}


def get_gcp_client():
    private_key = os.getenv("PRIVATE_KEY")
    client_email = os.getenv("CLIENT_EMAIL")
    project_id = os.getenv("PROJECT_ID")

    if not private_key or not client_email or not project_id:
        raise ValueError("Missing credentials. Please check your .env file.")

    credentials = service_account.Credentials.from_service_account_info(
        {
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": os.getenv("PRIVATE_KEY_ID"),
            "private_key": private_key,
            "client_email": client_email,
            "client_id": os.getenv("CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
        }
    )
    return storage.Client(credentials=credentials, project=project_id)


@app.delete("/delete-image", tags=["Wedding Operations"])
async def delete_image(
    image_url: Optional[str] = Header(
        None, description="URL of the image to be deleted"
    ),
    image_id: Optional[str] = Header(
        None,
        description="ID of the image to be deleted (either image_id or image_url should be provided)",
    ),
    bearer_token: str = Header(...),
    wedding_session_id: Optional[str] = Header(
        None, description="(Optional) existing wedding-session ID"
    ),
):
    # Check that either image_id or image_url is provided
    if not image_url and not image_id:
        raise HTTPException(
            status_code=400, detail="Either image_id or image_url must be provided"
        )

    if not wedding_session_id:
        raise HTTPException(
            status_code=400,
            detail="Missing wedding_session_id")

    user_id, sess_oid = auth_and_get_ids(bearer_token, wedding_session_id)

    # --------------- Case 1: image_url provided ---------------
    key = None
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    if image_id:
        print(f"[DEBUG] Received image_id: {image_id}")

        # We will try to reconstruct the exact object key (including real extension)
        # by looking up the stored URL in DB. Fall back to a sensible default.
        image_id_without_extension = image_id.split(".")[0]
        print(
            f"[DEBUG] Extracted image_id_without_extension: {image_id_without_extension}"
        )

        # Prepare DB filter early to reuse below
        filter_query = {
            "bearer_token": bearer_token,
            "wedding_session_id": sess_oid}
        print(f"[DEBUG] Using filter_query for DB lookup: {filter_query}")

        # Attempt to find this image's URL from the user's document
        document = db["user_image_urls"].find_one(filter_query)
        print(f"[DEBUG] Document found: {bool(document)}")

        found_url = None
        if document:
            # Look in legacy flat images array
            imgs = document.get("images", [])
            print(
                f"[DEBUG] Checking legacy 'images' array with {
                    len(imgs)} entries...")
            for img in imgs:
                print(
                    f"[DEBUG] Checking image: {
                        img.get('image_id')} -> {image_id_without_extension}"
                )
                if img.get("image_id") == image_id_without_extension:
                    found_url = img.get("url")
                    print(
                        f"[DEBUG] Found matching image in legacy array: {found_url}")
                    break

            # If not found, search new images_by_category structure
            if not found_url and isinstance(
                    document.get("images_by_category"), dict):
                print("[DEBUG] Searching in 'images_by_category' structure...")
                for _cat, img_list in document["images_by_category"].items():
                    print(
                        f"[DEBUG] Checking category '{_cat}' with {
                            len(img_list)} items..."
                    )
                    for img in img_list:
                        print(
                            f"[DEBUG] Checking image: {
                                img.get('image_id')} -> {image_id_without_extension}"
                        )
                        if img.get("image_id") == image_id_without_extension:
                            found_url = img.get("url")
                            print(
                                f"[DEBUG] Found matching image in category '{_cat}': {found_url}"
                            )
                            break
                    if found_url:
                        break

        if found_url:
            image_url = found_url
            print(f"[DEBUG] ✅ Reconstructed image_url from DB: {image_url}")
        else:
            print("[DEBUG] ⚠️ No matching image URL found in DB; using fallback logic.")

    # If key not provided explicitly, reconstruct from URL or fallback
    use_s3 = False
    s3_bucket_for_delete = None
    print(
        f"[DEBUG] use_s3 set to: {use_s3}, s3_bucket_for_delete: {s3_bucket_for_delete}"
    )

    if image_url:
        try:
            # Example:
            # https://storage.googleapis.com/my-bucket/test/myimage.jpg
            if "storage.googleapis.com" in image_url:
                parts = image_url.split("/")
                bucket_name = parts[3]
                key = "/".join(parts[4:])
            elif ".storage.googleapis.com" in image_url:
                # Example:
                # https://my-bucket.storage.googleapis.com/test/myimage.jpg
                parts = image_url.split("/")
                bucket_name = parts[2].split(".")[0]
                key = "/".join(parts[3:])
            else:
                raise ValueError("Invalid GCP image URL format")

            # image_id is filename without extension
            image_id = key.split("/")[-1]
            print("bucket_name:", bucket_name)
            print("key:", key)
            print("image_id:", image_id)

        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid image URL format")
    else:
        raise HTTPException(500, f"Image Url not found")

    # --------------- Case 2: Only image_id provided ---------------
    if not image_id:
        raise HTTPException(status_code=400, detail="Missing image_id")

    if not key:
        # Rebuild GCP object key (adjust if you always upload into `test/`)
        key = f"test/{image_id}.jpg"

    # 1) Delete from storage -------------------------------------------------
    try:
        storage_client = get_gcp_client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(key)
        blob.delete()  # delete file from bucket
    except Exception as e:
        raise HTTPException(500, f"Failed to delete image from storage: {e}")

    # 2) Remove from database ------------------------------------------------

    if not document:
        raise HTTPException(
            status_code=404, detail="No matching document found for user/session"
        )

    modified = False

    # --- Clean up "images" array ---
    if "images" in document:
        new_images = [
            img
            for img in document["images"]
            if img.get("image_id") != image_id_without_extension
        ]
        if len(new_images) != len(document["images"]):
            print(
                f"Removing image_id {image_id_without_extension} from 'images' array")
            document["images"] = new_images
            modified = True

    # --- Clean up "images_by_category" arrays ---
    if "images_by_category" in document:
        for category, img_list in document["images_by_category"].items():
            new_img_list = [
                img
                for img in img_list
                if img.get("image_id") != image_id_without_extension
            ]
            if len(new_img_list) != len(img_list):
                print(
                    f"Removing image_id {image_id_without_extension} from category '{category}'"
                )
                document["images_by_category"][category] = new_img_list
                modified = True

    if modified:
        print(f"Document modified: {document['_id']}")
        document["updated_at"] = now_ist
        db["user_image_urls"].replace_one({"_id": document["_id"]}, document)

    return {"message": "Image deleted successfully", "image_id": image_id}


@app.get("/images", tags=["Wedding Operations"])
async def list_images(
    bearer_token: Optional[str] = Header(
        None, description="Bearer token for authentication"
    ),
    description: Optional[str] = Query(
        None,
        description="Keyword/phrase to match image descriptions (case-insensitive substring)",
    ),
    session_id: Optional[str] = Query(
        None, description="Wedding session ID (optional)"
    ),
):
    # --- Validate bearer_token
    if not bearer_token:
        return JSONResponse(
            status_code=401,
            content={
                "status": "error",
                "message": "Missing bearer_token header."},
        )

    # Lookup user for bearer_token
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user or "_id" not in user:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid bearer token."},
        )
    user_id = str(user["_id"])

    # --- Build query based on provided parameters
    if session_id:
        # Case: User wants to see images from a specific wedding session
        # First, verify they have access to this session
        try:
            _auth_session(
                bearer_token=bearer_token,
                wedding_session_id=session_id)
            # If we’re here, auth passed; convert to ObjectId for filtering
            sess_oid = ObjectId(session_id)
            query = {
                "wedding_session_id": sess_oid
            }  # Use only session_id to fetch records
        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"status": "error", "message": str(e.detail)},
            )
        except Exception:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Not authorized to access this wedding session",
                },
            )

        # If authorized, get ALL images uploaded to this session (from any
        # user)
        sess_oid = ObjectId(session_id)
        query = {"wedding_session_id": sess_oid}

    else:
        # Case: User wants to see their personal images (no session)
        # Return only images uploaded by this specific user without any session
        query = {
            "bearer_token": bearer_token,
            "$or": [
                {"wedding_session_id": {"$exists": False}},
                {"wedding_session_id": None},
                {"wedding_session_id": ""},
            ],
        }

    # --- Fetch documents matching the query
    cursor = db["user_image_urls"].find(query, {"_id": 0, "images": 1})

    all_images = []
    for doc in cursor:
        all_images.extend(doc.get("images", []))

    # --- Optional description filter: case-insensitive substring match
    if description:
        needle = description.strip().lower()
        all_images = [
            img
            for img in all_images
            if needle == (img.get("description", "") or "").lower()
        ]

    # --- Shape/whitelist fields to avoid exposing anything sensitive
    result = [
        {
            "description": img.get("description"),
            "url": img.get("url"),
            "uploaded_at": img.get("uploaded_at"),
            "image_id": img.get("image_id"),
            "category": img.get("category"),
        }
        for img in all_images
    ]

    # Always succeed with an array (possibly empty)
    return {"status": "success", "data": result}


# ================================================================================================================


@app.post("/user-info", tags=["User Operations"])
async def update_user_info(
    user_info: UserInfo,
    bearer_token: str = Header(..., description="Your existing bearer token"),
):
    user_col = get_collection("userOperations")

    # 1) Must already exist
    existing = user_col.find_one({"bearer_token": bearer_token})
    if not existing:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    # 2) Perform in‐place update
    update_fields = user_info.dict(exclude_unset=True)
    user_col.update_one({"_id": existing["_id"]}, {"$set": update_fields})

    # 3) Return the same token, so nothing ever changes
    return {
        "message": "User information saved successfully",
        "bearer_token": bearer_token,
    }


@app.post("/notification-fcm-token", tags=["User Operations"])
async def update_fcm_token(
    fcm_data: FCMTokenUpdate,
    bearer_token: str = Header(..., description="Your existing bearer token"),
):
    user_col = get_collection("userOperations")

    # 1) Must already exist
    existing = user_col.find_one({"bearer_token": bearer_token})
    if not existing:
        raise HTTPException(status_code=200, detail="Invalid bearer token")

    # 2) Update only the FCM token
    user_col.update_one(
        {"_id": existing["_id"]}, {"$set": {"fcm_token": fcm_data.fcm_token}}
    )

    # 3) Return the same token
    return {"message": "FCM token updated successfully",
            "bearer_token": bearer_token}


@app.get("/user-info", tags=["User Operations"])
async def get_user_info(
    bearer_token: str = Header(..., description="Your existing bearer token")
):
    user_col = get_collection("userOperations")
    user_data = user_col.find_one({"bearer_token": bearer_token})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = convert_object_ids(user_data)
    return user_data


@app.post("/wedding-info")
async def create_or_update_wedding_info(
    wedding_info: WeddingInfo,
    bearer_token: str = Header(...),
    session_id: Optional[str] = Query(
        None, description="(Optional) existing wedding-session ID to update"
    ),
):
    # 1) Authenticate caller
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    caller_id = user["_id"]

    coll = db["weddingSessionOperations"]

    print("caller_id", caller_id)
    print("session_id", session_id)

    if wedding_info.name:
        # Check current name value
        current_name = user.get("name")
        if not current_name or current_name == "name":
            db["userOperations"].update_one(
                {"bearer_token": bearer_token}, {
                    "$set": {"name": wedding_info.name}}
            )

    # 2) If client gave us a session_id, update that document
    if session_id:
        try:
            oid = ObjectId(session_id)
        except (bson_errors.InvalidId, TypeError):
            raise HTTPException(status_code=400, detail="Malformed session_id")

        existing = coll.find_one({"_id": oid})
        if not existing:
            raise HTTPException(status_code=404, detail="Session not found")

        # build our $set payload
        set_payload: Dict[str, Any] = wedding_info.dict(exclude_unset=True)

        # only stamp users.user_id if updater is not the original bride
        bride_id = existing.get("users", {}).get("bride")
        if caller_id != bride_id:
            set_payload["users.user_id"] = caller_id

        # Generate and add referral token from the last 6 digits of the session
        # _id
        referral_token = str(oid)[-6:]
        set_payload["referralToken"] = referral_token

        coll.update_one({"_id": oid}, {"$set": set_payload})

        return {
            "message": "Wedding information updated successfully",
            "session_id": session_id,
            "referralToken": referral_token,
        }

    # 3) Otherwise fall back to create-or-update by bride
    data = wedding_info.dict()
    data["users"] = {"bride": caller_id}

    result = coll.insert_one(data)
    new_id = str(result.inserted_id)

    # Generate referral token from the last 6 digits of the newly created _id
    referral_token = str(result.inserted_id)[-6:]

    # Update the document with the referral token
    coll.update_one(
        {"_id": result.inserted_id}, {"$set": {"referralToken": referral_token}}
    )

    return {
        "message": "New wedding session created successfully",
        "session_id": new_id,
        "referralToken": referral_token,
    }


@app.get("/wedding-info")
async def get_wedding_info(
    *,
    bearer_token: str = Header(..., description="Your user’s bearer token"),
    session_id: str = Query(
        ..., description="The wedding-session ID (ObjectId) to fetch"
    ),
):
    # 1) Authenticate caller
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    caller_id = user["_id"]

    coll = db["weddingSessionOperations"]

    # 2) Validate & convert session_id → ObjectId
    try:
        oid = ObjectId(session_id)
    except (bson_errors.InvalidId, TypeError):
        raise HTTPException(status_code=400, detail="Malformed session_id")

    # 3) Fetch the exact session document
    doc = coll.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Session not found")

    # 4) Authorization: must be bride or in participants (inside users)
    users = doc.get("users", {})
    participants = users.get("participants", {})

    is_bride = caller_id == users.get("bride")
    is_participant = str(caller_id) in [str(uid)
                                        for uid in participants.values()]

    print(
        f"Caller ID: {caller_id}, Bride ID: {
            users.get('bride')}, Participants: {participants}"
    )

    if not (is_bride or is_participant):
        raise HTTPException(
            status_code=403, detail="Not authorized to view this session"
        )

    # 5) Convert every ObjectId in the document into a string
    doc["session_id"] = str(doc["_id"])
    del doc["_id"]

    # 6) JSON-encodable version
    safe_doc = jsonable_encoder(doc, custom_encoder={ObjectId: str})

    return JSONResponse(content=safe_doc)


@app.get("/user-sessions")
async def get_user_sessions(
    bearer_token: str = Header(..., description="Your user’s bearer token")
):
    """
    List every wedding-session this user participates in,
    with their role ("bride" or "participant") and the session's ID and wedding name.
    """
    # 1) Authenticate caller
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    caller_id = user["_id"]
    caller_username = user.get("name", "")

    coll = db["weddingSessionOperations"]

    cursor = coll.find(
        {
            "$or": [
                {"users.bride": caller_id},  # User is the bride
                {f"users.participants.{caller_username}": str(caller_id)},
                {"users.participants": {"$in": [str(caller_id)]}},
            ]
        }
    )

    sessions = []
    for doc in cursor:
        users = doc.get("users", {})
        participants = users.get("participants", {})

        # Determine role
        if users.get("bride") == caller_id:
            role = "bride"
        elif str(caller_id) in participants.values() or participants.get(
            caller_username
        ) == str(caller_id):
            role = "participant"
        else:
            # shouldn't happen
            role = "unknown"
        sessions.append(
            {
                "session_id": str(doc["_id"]),
                "role": role,
                "wedding_name": doc.get("wedding_name", ""),
            }
        )

    return {"sessions": sessions}


# @app.post("/email/send", tags=["Contact Operations"])
# async def email_send(
#     background_tasks: BackgroundTasks,
#     to_email: str = Form(..., example="friend@example.com"),
#     subject: str  = Form(..., example="Hello from SayYesAI"),
#     body: str     = Form(..., example="Congrats on your engagement!"),
# ):
#     """
#     Queues an e-mail via BackgroundTasks so the HTTP response is instant.
#     """
#     background_tasks.add_task(
#         send_email_async,
#         to_email,
#         subject,
#         html_body=f"<p>{body}</p>",
#         text_body=body,
#     )
#     return {"detail": "Queued ✅"}


@app.post("/send_whatsapp_message", tags=["Contact Operations"])
async def send_whatsapp_template(
    phone_number: str, user_id: str = Header(...), session_id: str = Header(...)
):
    """
    Send a welcome template to the user's phone number.
    """
    print(f"Sending welcome message to {phone_number}")
    url = "https://api.toingg.com/api/v3/send_whatsapp_template"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer psdTo0QFiErjJ9SlAG0VhI4RkKq7nTNOhmClw0xc6B5Vl5z7heCvZ6C9eW01KGgM",
    }
    params = {
        "templateName": "say_yes_demo",
        "langCode": "en",
        "phoneNumber": phone_number,
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            print(f"Response from Toingg: {response.status} {await response.text()}")
            if response.status != 200:
                raise HTTPException(status_code=400, detail=await response.text())
            return JSONResponse(
                content={"message": "Message sent successfully"}, status_code=200
            )


# -----------------------------------
# Global dictionaries to track idle state and conversation state per user
# -----------------------------------
responses_dict: Dict[str, Dict[str, Any]] = {}

# -----------------------------------
# Chat Segmentation Endpoints
# -----------------------------------


@app.get("/chat/segment_new/{segment}", tags=["Chat Segments"])
async def get_segment_chat_with_recommendations(
    segment: str,
    bearer_token: str = Header(...),
    wedding_session_id: str = Header(...),
    request: Request = None,
):
    user = authenticate_user(bearer_token)
    user_id = str(user["_id"])
    # print("segement", segment)
    chats = await get_segment_chat(user_id, wedding_session_id, segment)
    # print("chats", chats)
    updated_chats = []
    # print("chats", chats)
    for chat in chats:
        # print("segement------------------------------------1", segment)

        role = chat.get("role")
        content = chat.get("content", "")

        # Skip appending if content contains 'Invite_Party' (case-insensitive), starts from user with 'START',
        # or is a dict with ResponseComplete.type == 'Invite_Party'
        skip_invite_party = False
        if isinstance(content, str):
            if re.search(r"invite[_\s]?party", content, re.IGNORECASE) or (
                role == "user" and content.strip().upper().startswith("START")
            ):
                skip_invite_party = True
        elif isinstance(content, dict):
            # Check for the nested Invite_Party type in ResponseComplete
            response_complete = content.get("ResponseComplete")
            if (
                isinstance(response_complete, dict)
                and response_complete.get("type", "").lower() == "invite_party"
            ):
                skip_invite_party = True
        if skip_invite_party:
            continue

        # print("segement--------------------------------------2", segment)

        # print("Keys of content:", content.keys())
        # print("Type of content:", type(content))
        # print("Content:", content)

        if (
            isinstance(content, str) and "recommendations_uuid:" in content
        ):  # Extract values using regex
            # print("enterred in recommendations_uuid")
            uuid_match = re.search(
                r"recommendations_uuid:\s*([\w-]+)", content)
            session_id_match = re.search(r"session_id:\s*([\w-]+)", content)
            bearer_token_match = re.search(
                r"bearer_token:\s*([\w-]+)", content)
            print("uuid_match", uuid_match)

            def get_recommendation_by_uuid(
                session_id: str, bearer_token: str, recommendation_uuid: str
            ):
                collection = db["chat_recommendations"]
                print("session_id", session_id)
                print("bearer_token", bearer_token)
                print("recommendation_uuid", recommendation_uuid)
                # 1. Find user by bearer_token
                user_doc = db["userOperations"].find_one(
                    {"bearer_token": bearer_token})
                if not user_doc or "_id" not in user_doc:
                    return {"in valid bearer token": bearer_token}
                user_obj_id = user_doc["_id"]

                # 2. Find recommendations for this user/session
                doc = collection.find_one(
                    {
                        "user_id": user_obj_id,
                        "session_id": session_id,
                    }
                )

                if not doc:
                    return {
                        "error": "No recommendations found for this user/session."}

                recs = doc.get("recommendations", {})
                if not recs:
                    return {"error": "No recommendations found."}

                for vendor_cat, vendor_data in recs.items():
                    if (
                        isinstance(vendor_data, dict)
                        and recommendation_uuid in vendor_data
                    ):
                        uuid_data = vendor_data[recommendation_uuid]
                        # Clean the data to remove NaN/inf and convert ObjectId
                        # to str
                        safe_data = _clean_bson(uuid_data)
                        return {
                            "recommendations": safe_data,
                        }

                return {"error": "Recommendation UUID not found."}

            content = get_recommendation_by_uuid(
                session_id=wedding_session_id,
                bearer_token=bearer_token,
                recommendation_uuid=uuid_match.group(
                    1) if uuid_match else None,
            )

            updated_chats.append(
                {
                    "type": "recommendation_agent",
                    "content": json.dumps({"Artifact": content}),
                    "segment": segment,
                }
            )
            continue
        # Keep original chat if no replacement happened
        updated_chats.append(chat)
    # print("segement----------------------------3", segment)

    return updated_chats


@app.get("/chat/segment/{segment}", tags=["Chat Segments"])
async def get_chats_for_segment(
    segment: str,
    user_id: str = Query(None, description="User ID"),
    bearer_token: str = Header(None, description="Bearer Token"),
    wedding_session_id: str = Query(..., description="Wedding Session ID"),
):
    """Get all chat messages for a given segment."""

    if bearer_token:
        user = db["userOperations"].find_one({"bearer_token": bearer_token})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        user_id = str(user["_id"])  # Use _id from the document as user_id

    if not user_id:
        raise HTTPException(
            status_code=400, detail="Either user_id or bearer_token must be provided"
        )

    # Proceed to get chat messages for the segment
    chats = await get_segment_chats(user_id, wedding_session_id, segment)
    return {"chats": chats}


@app.get("/chat/segments", tags=["Chat Segments"])
async def list_chat_segments(
    user_id: str = Query(None, description="User ID"),
    bearer_token: str = Header(None, description="Bearer Token"),
    wedding_session_id: str = Query(..., description="Wedding Session ID"),
):
    """List all segment names for a user's session."""

    if bearer_token:
        user = db["userOperations"].find_one({"bearer_token": bearer_token})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid bearer token")
        user_id = str(user["_id"])  # Use _id from the document as user_id

    if not user_id:
        if not user_id:
            raise HTTPException(
                status_code=400,
                detail="Either user_id or bearer_token must be provided",
            )

    # Proceed to get segments for the user with the given wedding session id
    segments = await get_segments_for_user(user_id, wedding_session_id)
    return {"segments": segments}


@app.post("/add_vendor_data", tags=["Budget"])
async def add_vendor_data(
    vendor_name: str = Query(..., description="Vendor Name"),
    session_id: str = Query(..., description="Session ID"),
    bearer_token: str = Header(..., description="Bearer Token"),
    details: Dict[str, Dict[str, str]
                  ] = Body(..., description="Vendor Details"),
):
    """
    Adds or updates vendor data associated with a session and bearer token.
    If no existing record is found, a new record is created.
    If an existing record is found, the new vendor name is appended.
    """

    # Authenticate user based on bearer token
    user = authenticate_user(bearer_token)

    budget_collection = db["budget"]

    # Search for an existing record based on session and bearer token
    existing_record = budget_collection.find_one(
        {"session_id": session_id, "bearer_token": bearer_token}
    )

    if not existing_record:
        # If no existing record, create a new record with vendor data
        new_record = {
            "session_id": session_id,
            "bearer_token": bearer_token,
            "vendors": {vendor_name: details},
        }
        # Insert the new record into MongoDB
        budget_collection.insert_one(new_record)
    else:
        # If an existing record, update the vendors field
        vendors = existing_record.get("vendors", {})

        if vendor_name in vendors:
            # If the vendor already exists, update the details
            vendors[vendor_name].update(details)
        else:
            # If the vendor doesn't exist, add a new entry
            vendors[vendor_name] = details

        # Update the existing record with the modified vendors field
        budget_collection.update_one(
            {"_id": existing_record["_id"]}, {"$set": {"vendors": vendors}}
        )

    return {"status": "success", "message": "Vendor data added successfully"}


@app.get("/get_vendor_data", tags=["Budget"])
async def get_vendor_data(
    session_id: str = Header(..., description="Session ID"),
    bearer_token: str = Header(..., description="Bearer Token"),
):
    """
    Fetches all records associated with the provided session and bearer token.
    """

    # Authenticate user based on bearer token
    user = authenticate_user(bearer_token)

    budget_collection = db["budget"]
    # Search for existing record using session_id and bearer_token
    existing_record = budget_collection.find_one(
        {"session_id": session_id, "bearer_token": bearer_token}
    )

    if not existing_record:
        raise HTTPException(
            status_code=404, detail="Record not found for the given session and token"
        )

    # Fetch all records from the collection
    all_records = list(
        budget_collection.find(
            {"session_id": session_id, "bearer_token": bearer_token})
    )

    # Remove the MongoDB internal _id field from the response
    for record in all_records:
        del record["_id"]  # Remove the ObjectId
        # Optionally remove bearer_token from response
        del record["bearer_token"]
        del record["session_id"]  # Optionally remove session_id from response

    return {"status": "success",
            "message": "Fetched all records", "data": all_records}


# Webhook receiver endpoint for your hosted app
@app.post("/webhook/receive/{user_id}", tags=["Webhooks"])
async def receive_webhook_notification(
        user_id: str, webhook_data: dict = Body(...)):
    """
    Receive webhook notifications for specific users.
    This runs on your hosted backend: https://sayyesai-backend-984515.onrender.com
    """
    try:
        print(f"Received webhook for user {user_id}: {webhook_data}")

        event_type = webhook_data.get("event")
        wedding_session_id = webhook_data.get("wedding_session_id")

        if event_type == "member_joined":
            joiner_name = webhook_data.get("joiner_name")
            message = webhook_data.get("message")

            # Store the notification in database for the user to fetch
            notification_doc = {
                "user_id": user_id,
                "wedding_session_id": wedding_session_id,
                "event_type": event_type,
                "message": message,
                "joiner_name": joiner_name,
                "created_at": datetime.utcnow(),
                "is_read": False,
                "webhook_received": True,
            }

            db["webhook_notifications"].insert_one(notification_doc)
            print(f"Stored webhook notification for user {user_id}")

            # If the user (bride) has a WebSocket connection, send real-time
            # notification
            session_key = f"{user_id}_{wedding_session_id}"
            if session_key in user_websockets:
                try:
                    websocket = user_websockets[session_key]
                    await websocket.send_text(
                        json.dumps(
                            {"type": "webhook_notification", "data": webhook_data}
                        )
                    )
                    print(f"Sent real-time notification to user {user_id}")
                except Exception as e:
                    print(f"Failed to send WebSocket notification: {e}")

        return {"status": "received", "user_id": user_id}

    except Exception as e:
        print(f"Error processing webhook for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Webhook processing failed")


# API to fetch webhook notifications for users
@app.get("/webhook/notifications", tags=["Webhooks"])
async def get_webhook_notifications(
    bearer_token: str = Header(...,
                               description="Bearer token for authentication"),
    wedding_session_id: str = Query(..., description="Wedding session ID"),
    unread_only: bool = Query(
        True, description="Fetch only unread notifications"),
):
    """
    Get webhook notifications for the authenticated user.
    Users call this to check for new webhook notifications.
    """
    # Authenticate user
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = str(user["_id"])

    # Build query
    query = {"user_id": user_id, "wedding_session_id": wedding_session_id}

    if unread_only:
        query["is_read"] = False

    # Fetch notifications
    notifications = list(
        db["webhook_notifications"].find(
            query).sort("created_at", -1).limit(20)
    )

    # Convert ObjectId to string
    for notification in notifications:
        notification["_id"] = str(notification["_id"])

    return {"notifications": notifications, "count": len(notifications)}


# Mark webhook notifications as read
@app.post("/webhook/notifications/mark-read", tags=["Webhooks"])
async def mark_webhook_notifications_read(
    bearer_token: str = Header(...,
                               description="Bearer token for authentication"),
    notification_ids: List[str] = Body(
        ..., description="List of notification IDs to mark as read"
    ),
):
    """Mark webhook notifications as read."""
    user = db["userOperations"].find_one({"bearer_token": bearer_token})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = str(user["_id"])

    # Convert to ObjectIds
    try:
        object_ids = [ObjectId(nid) for nid in notification_ids]
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid notification IDs")

    # Update notifications
    result = db["webhook_notifications"].update_many(
        {"_id": {"$in": object_ids}, "user_id": user_id},
        {"$set": {"is_read": True, "read_at": datetime.utcnow()}},
    )

    return {
        "message": f"Marked {result.modified_count} notifications as read",
        "modified_count": result.modified_count,
    }


# Update the join endpoint to trigger webhooks
@app.post("/wedding-session/join", tags=["Wedding Operations"])
async def join_wedding_session(
    session_id: str = Body(
        ..., embed=True, description="ObjectId of the weddingSessionOperations document"
    ),
    bearer_token: str = Header(..., description="Your existing bearer token"),
):
    try:
        # Get user information first
        user = db["userOperations"].find_one({"bearer_token": bearer_token})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid bearer token")

        user_id = user.get("_id")
        username = user.get("name", "Unknown User")

        coll = db["weddingSessionOperations"]
        session = None
        sess_oid = None
        actual_session_id = session_id  # Keep original for response

        # Determine if session_id is an ObjectId or referral token
        # Try referral token first if it looks like one (6 characters)
        if len(session_id) == 6 and session_id.isalnum():
            # Try to find by referral token
            session = coll.find_one({"referralToken": session_id})
            if session:
                sess_oid = session["_id"]
                actual_session_id = str(
                    sess_oid
                )  # Convert to ObjectId string for consistent handling

        # Check for invite-based join first
        inv = db["userOperations"].find_one(
            {"bearer_token": bearer_token, "user_id": None}
        )
        # if not sess_oid:
        #     try:
        #         invite_session_oid = ObjectId(session_id)
        #     except (bson_errors.InvalidId, TypeError):
        #         raise HTTPException(status_code=400, detail="Invalid session_id")
        # else:
        #     invite_session_oid = sess_oid

        if inv:
            # Handle invite-based join
            db["userOperations"].update_one(
                {"_id": inv["_id"]}, {"$set": {"user_id": user_id}}
            )

            if not sess_oid:
                try:
                    invite_session_oid = ObjectId(session_id)
                except (bson_errors.InvalidId, TypeError):
                    raise HTTPException(
                        status_code=400, detail="Invalid session_id")
            else:
                invite_session_oid = sess_oid

            # Ensure referral token exists
            if not session:
                session = coll.find_one({"_id": invite_session_oid})

            referral_token = session.get("referralToken") if session else None
            if not referral_token:
                referral_token = str(invite_session_oid)[-6:]

            # Add user to the participants map and update referral token
            update_result = coll.update_one(
                {"_id": invite_session_oid},
                {
                    "$set": {
                        f"users.participants.{username}": str(user_id),
                        "referralToken": referral_token,
                    }
                },
            )

            if update_result.matched_count == 0:
                raise HTTPException(
                    status_code=404, detail="Session not found in invite flow"
                )

            # Seed 5 daily image-upload tasks for this new participant
            # (idempotent)
            try:
                session_id_str = str(invite_session_oid)

                # Resolve bride's bearer token for creator field
                bride_id = session.get("users", {}).get(
                    "bride") if session else None
                bride_doc = None
                try:
                    if bride_id is not None:
                        bride_doc = db["userOperations"].find_one(
                            {"_id": bride_id})
                except Exception:
                    pass
                if not bride_doc and bride_id:
                    try:
                        bride_doc = db["userOperations"].find_one(
                            {"_id": ObjectId(str(bride_id))}
                        )
                    except Exception:
                        bride_doc = None

                bride_bearer_token = (
                    bride_doc.get("bearer_token") if bride_doc else None
                )

                task_names = [
                    "Most memorable pic with the bride",
                    "Funniest moment together",
                    "Best throwback outfit with the bride",
                    "Your favorite trip photo with the bride",
                    "A candid moment only you have",
                ]

                base_date = datetime.now(IST).date()
                created_count = 0
                for i, tname in enumerate(task_names):
                    assigned_date = (base_date + timedelta(days=i)).isoformat()
                    due_date = assigned_date

                    existing = db["wedding_tasks"].find_one(
                        {
                            "wedding_session_id": session_id_str,
                            "assignee_id": bearer_token,
                            "taskName": tname,
                        }
                    )
                    if existing:
                        continue

                    task_doc = {
                        "task_id": str(uuid4()),
                        "taskName": tname,
                        "taskStatus": "pending",
                        "bearer_token": bride_bearer_token or bearer_token,
                        "wedding_session_id": session_id_str,
                        "createdAt": datetime.utcnow(),
                        "assignee_id": bearer_token,
                        "taskDescription": "Please upload a photo for this prompt.",
                        "taskType": "image",
                        "assignedDate": assigned_date,
                        "dueDate": due_date,
                    }
                    db["wedding_tasks"].insert_one(task_doc)
                    created_count += 1

                print(
                    f"[join] Seeded {created_count} scheduled image-upload tasks for participant {username}"
                )
            except Exception as seed_err:
                print(
                    f"[join] Failed to seed scheduled tasks for participant {username}: {seed_err}"
                )

            # Send webhooks to all session members
            await send_webhook_to_session_members(
                wedding_session_id=str(invite_session_oid),
                joiner_name=username,
                joiner_user_id=str(user_id),
            )
            await send_member_joined_notification(
                wedding_session_id=actual_session_id,  # Correct variable
                joiner_name=username,
                joiner_user_id=str(user_id),
            )
            print(f"Sent member joined notifications for {username}")
            # Save to in_app_notifications collection
            print(f"notification to in app colletion inserting started in main.py")
            await notify_session_about_new_member(
                wedding_session_id=str(invite_session_oid),
                joiner_name=username,
                joiner_user_id=str(user_id),
            )
            print(
                f"Notification sent for {username} joining and saved to in app notifications colletion !!!"
            )
            print(f"Sent webhook notifications for {username} joining")

            return {
                "message": f"Joined via invite to {session.get('wedding_name', 'Wedding Session')}!",
                "session_id": actual_session_id,  # ✅ Use this instead
                "referralToken": referral_token,
                "wedding_name": session.get("wedding_name", "Wedding Session"),
            }

        # Handle regular join flow
        if not session:
            try:
                sess_oid = ObjectId(session_id)
                session = coll.find_one({"_id": sess_oid})
                actual_session_id = str(sess_oid)
            except (bson_errors.InvalidId, TypeError):
                raise HTTPException(
                    status_code=400, detail="Invalid session_id or referral token"
                )

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Wedding session not found")

        # Check if user is already the bride
        bride_id = session.get("users", {}).get("bride", "")
        if str(user_id) == str(bride_id):
            raise HTTPException(
                status_code=400,
                detail="You cannot join as a participant if you are the bride",
            )

        participants = session.get("users", {}).get("participants", {})
        if not isinstance(participants, dict):
            raise HTTPException(
                status_code=500, detail="Corrupted session participants data"
            )

        referral_token = session.get("referralToken")
        if not referral_token:
            referral_token = str(sess_oid)[-6:]

        if username in participants:
            if not session.get("referralToken"):
                coll.update_one(
                    {"_id": sess_oid}, {"$set": {"referralToken": referral_token}}
                )

            return {
                "message": f"You are already a participant in {session.get('wedding_name', 'Wedding Session')}",
                "referralToken": referral_token,
                "wedding_name": session.get("wedding_name", "Wedding Session"),
            }

        # Add to participants
        update_result = coll.update_one(
            {"_id": sess_oid},
            {
                "$set": {
                    f"users.participants.{username}": str(user_id),
                    "referralToken": referral_token,
                }
            },
        )

        if update_result.matched_count == 0:
            raise HTTPException(
                status_code=404, detail="Failed to update session (not found)"
            )

        # Seed 5 daily image-upload tasks for this new participant (idempotent)
        try:
            session_id_str = actual_session_id

            # Resolve bride's bearer token for creator field
            bride_id = session.get("users", {}).get(
                "bride") if session else None
            bride_doc = None
            try:
                if bride_id is not None:
                    bride_doc = db["userOperations"].find_one(
                        {"_id": bride_id})
            except Exception:
                pass
            if not bride_doc and bride_id:
                try:
                    bride_doc = db["userOperations"].find_one(
                        {"_id": ObjectId(str(bride_id))}
                    )
                except Exception:
                    bride_doc = None

            bride_bearer_token = bride_doc.get(
                "bearer_token") if bride_doc else None

            task_names = [
                "Most memorable pic with the bride",
                "Funniest moment together",
                "Best throwback outfit with the bride",
                "Your favorite trip photo with the bride",
                "A candid moment only you have",
            ]

            base_date = datetime.now(IST).date()
            created_count = 0
            for i, tname in enumerate(task_names):
                assigned_date = (base_date + timedelta(days=i)).isoformat()
                due_date = assigned_date

                existing = db["wedding_tasks"].find_one(
                    {
                        "wedding_session_id": session_id_str,
                        "assignee_id": bearer_token,
                        "taskName": tname,
                    }
                )
                if existing:
                    continue

                task_doc = {
                    "task_id": str(uuid4()),
                    "taskName": tname,
                    "taskStatus": "pending",
                    "bearer_token": bride_bearer_token or bearer_token,
                    "wedding_session_id": session_id_str,
                    "createdAt": datetime.utcnow(),
                    "assignee_id": bearer_token,
                    "taskDescription": "Please upload a photo for this prompt.",
                    "taskType": "image",
                    "assignedDate": assigned_date,
                    "dueDate": due_date,
                }
                db["wedding_tasks"].insert_one(task_doc)
                created_count += 1

            print(
                f"[join] Seeded {created_count} scheduled image-upload tasks for participant {username}"
            )
        except Exception as seed_err:
            print(
                f"[join] Failed to seed scheduled tasks for participant {username}: {seed_err}"
            )

        # Send webhooks to all session members
        await send_webhook_to_session_members(
            wedding_session_id=actual_session_id,
            joiner_name=username,
            joiner_user_id=str(user_id),
        )
        print(f"Sent webhook notifications for {username} joining")
        # In your join_wedding_session function, after successful join:
        # NEW:
        await send_member_joined_notification(
            wedding_session_id=actual_session_id,  # Correct variable
            joiner_name=username,
            joiner_user_id=str(user_id),
        )
        print(f"Sent member joined notifications for {username}")
        # Save to in_app_notifications collection
        await notify_session_about_new_member(
            wedding_session_id=actual_session_id,
            joiner_name=username,
            joiner_user_id=str(user_id),
        )
        return {
            "message": f"Successfully joined {session.get('wedding_name', 'Wedding Session')}",
            "session_id": actual_session_id,
            "referralToken": referral_token,
            "wedding_name": session.get("wedding_name", "Wedding Session"),
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print("[ERROR]", str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error while joining session"
        )


# Endpoint to remove a participant from wedding session (bride only)
@app.delete("/wedding-session/participant", tags=["Wedding Operations"])
async def remove_participant(
    session_id: str = Body(
        ..., embed=True, description="ObjectId of the weddingSessionOperations document"
    ),
    participant_bearer_token: str = Body(
        ..., embed=True, description="Bearer token of the participant to remove"
    ),
    bearer_token: str = Header(
        ..., description="Your existing bearer token (must be bride)"
    ),
):
    try:
        # Get user information first (bride)
        user = db["userOperations"].find_one({"bearer_token": bearer_token})
        if not user:
            raise HTTPException(status_code=401, detail="Invalid bearer token")

        user_id = user.get("_id")
        username = user.get("name", "Unknown User")

        # Get participant information from their bearer token
        participant_user = db["userOperations"].find_one(
            {"bearer_token": participant_bearer_token}
        )
        if not participant_user:
            raise HTTPException(
                status_code=404,
                detail="Participant not found with provided bearer token",
            )

        participant_user_id = participant_user.get("_id")
        participant_name = participant_user.get("name", "Unknown User")

        # Get session by ObjectId
        try:
            sess_oid = ObjectId(session_id)
        except (bson_errors.InvalidId, TypeError):
            raise HTTPException(status_code=400, detail="Invalid session_id")

        coll = db["weddingSessionOperations"]
        session = coll.find_one({"_id": sess_oid})

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Wedding session not found")

        # Check if the caller is the bride
        bride_id = session.get("users", {}).get("bride")
        if str(user_id) != str(bride_id):
            raise HTTPException(
                status_code=403, detail="Only the bride can remove participants"
            )

        # Get current participants
        participants = session.get("users", {}).get("participants", {})
        if not isinstance(participants, dict):
            raise HTTPException(
                status_code=500, detail="Corrupted session participants data"
            )

        # Find participant in session by user_id
        participant_found = False
        participant_name_in_session = None

        for name, stored_user_id in participants.items():
            if str(stored_user_id) == str(participant_user_id):
                participant_found = True
                participant_name_in_session = name
                break

        if not participant_found:
            raise HTTPException(
                status_code=404, detail="User is not a participant in this session"
            )

        # Check if bride is trying to remove themselves (not allowed)
        if str(participant_user_id) == str(bride_id):
            raise HTTPException(
                status_code=400,
                detail="Bride cannot remove themselves from the session",
            )

        # Remove the participant
        update_result = coll.update_one(
            {"_id": sess_oid},
            {"$unset": {f"users.participants.{participant_name_in_session}": ""}},
        )

        if update_result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Failed to update session")

        # Send webhook notifications to remaining session members about
        # participant removal
        await send_webhook_to_session_members(
            wedding_session_id=str(sess_oid),
            joiner_name=participant_name_in_session,
            joiner_user_id=str(participant_user_id),
        )
        print(
            f"Sent webhook notifications for {participant_name_in_session} being removed from session"
        )

        return {
            "message": f"Successfully removed {participant_name} from {session.get('wedding_name', 'Wedding Session')}",
            "session_id": str(sess_oid),
            "referralToken": session.get("referralToken", str(sess_oid)[-6:]),
            "wedding_name": session.get("wedding_name", "Wedding Session"),
            "removed_participant": participant_name,
            "removed_participant_id": str(participant_user_id),
            "removed_by": username,
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print("[ERROR]", str(e))
        raise HTTPException(
            status_code=500, detail="Internal server error while removing participant"
        )


async def get_user_liked_venues(
    user_id: str, bearer_token: str, wedding_session_id: str
):
    """
    Get venues that this user has previously liked during onboarding
    """
    try:
        print(
            f"DEBUG: Checking liked venues for user_id: {user_id}, bearer_token: {bearer_token}, wedding_session_id: {wedding_session_id}"
        )

        collection = db["onboarding_recommendations"]

        # First, let's check if there are any documents with liked_by_users
        # field at all
        total_venues = collection.count_documents({})
        venues_with_likes = collection.count_documents(
            {"liked_by_users": {"$exists": True}}
        )

        print(f"DEBUG: Total venues in collection: {total_venues}")
        print(f"DEBUG: Venues with likes: {venues_with_likes}")

        # Let's see what documents exist with liked_by_users
        sample_docs = list(
            collection.find(
                {"liked_by_users": {"$exists": True}}, {
                    "title": 1, "liked_by_users": 1}
            ).limit(3)
        )

        print(f"DEBUG: Sample documents with likes: {sample_docs}")

        # Find all venues where this user is in the liked_by_users array
        query = {
            "liked_by_users": {
                "$elemMatch": {
                    "user_id": user_id,
                    "bearer_token": bearer_token,
                    "wedding_session_id": wedding_session_id,
                }
            }
        }

        print(f"DEBUG: Query: {query}")

        liked_venues = list(
            collection.find(
                query,
                {
                    "_id": 0,
                    "title": 1,
                    "recommendation_summary": 1,
                    "liked_by_users": 1,
                },
            )
        )

        print(f"DEBUG: Found {len(liked_venues)} liked venues")
        print(f"DEBUG: Liked venues: {liked_venues}")

        return {"success": True, "data": liked_venues,
                "count": len(liked_venues)}

    except Exception as e:
        print(f"ERROR in get_user_liked_venues: {e}")
        return {"success": False, "error": str(e), "data": []}


# -----------------------------------
# WebSocket endpoint
# -----------------------------------

# Global dictionary to hold wedding info per user
# wedding_info_global: Dict[str, WeddingInfo] = {}


@app.websocket("/ws")
# Use below line to send it in params
# sample ws link: wss://your-server.com/ws?bearer-token=<enter your bearer-token here>
# async def websocket_endpoint(websocket: WebSocket, bearer_token: str =
# Query(..., alias="bearer-token")):
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint to accept client connections.
    Handles:
      - User authentication by bearer_token
      - Conversation history tracking
      - User idle checking (via idle_watcher task)
      - Dynamic watcher events integration
      - Global wedding info storage per user
    """
    await websocket.accept()

    """
    # try:
    #     # Get the headers from the WebSocket
    #     headers = dict(websocket.scope.get("headers", []))

    #     # Get bearer token from headers
    #     auth_header = next((v.decode() for k, v in headers.items() if k.lower() == b'bearer_token'), None)
    #     if not auth_header or not auth_header.startswith("Bearer "):
    #         logger.error("Missing or invalid authorization header")
    #         await websocket.close()
    #         return

    #     bearer_token = auth_header.split(" ")[1]

    #     # Get wedding session ID from headers
    #     wedding_session_id = next((v.decode() for k, v in headers.items() if k.lower() == b'wedding_session_id'), None)

    #     user_data = db["userOperations"].find_one({"bearer_token": bearer_token})
    #     if not user_data:
    #         logger.error(f"Invalid bearer token: {bearer_token}")
    #         await websocket.close()
    #         return

    #     user_id = user_data["_id"]
    #     logger.info(f"New WebSocket connection from user: {user_id}")
    #     user_last_seen[user_id] = datetime.now()

    #     print("bearer_token: ==========================================================================", bearer_token)
    #     print("wedding_session_id: ==========================================================================", wedding_session_id)

    # except Exception as e:
    #     logger.error(f"WebSocket error: {str(e)}")
    #     await websocket.close()
    """

    # Track the WebSocket connection in `user_websockets`

    try:
        # Get and log all headers exactly as received
        headers = dict(websocket.scope.get("headers", []))
        print("=== RAW HEADERS ===")
        for k, v in headers.items():
            print(f"Header: {k!r} -> {v!r}")
        print("===================")

        # Try to get bearer token with different possible header names
        bearer_token = None
        for header_name in [b"bearer_token",
                            b"authorization", b"bearer-token"]:
            if header_name in {k.lower(): v for k, v in headers.items()}:
                value = next(v for k, v in headers.items()
                             if k.lower() == header_name)
                value = value.decode("utf-8")
                if header_name == b"authorization" and value.lower().startswith(
                    "bearer "
                ):
                    bearer_token = value.split(" ")[1]
                else:
                    bearer_token = value
                break

        if not bearer_token:
            logger.error("No valid bearer token found in headers")
            await websocket.close()
            return

        # Get wedding session ID
        wedding_session_id = None
        for header_name in [b"wedding_session_id", b"wedding-session-id"]:
            if header_name in {k.lower(): v for k, v in headers.items()}:
                value = next(v for k, v in headers.items()
                             if k.lower() == header_name)
                wedding_session_id = value.decode("utf-8")
                break

        print(f"Extracted bearer_token: {bearer_token}")
        print(f"Extracted wedding_session_id: {wedding_session_id}")

        # Rest of your authentication logic...
        user_data = db["userOperations"].find_one(
            {"bearer_token": bearer_token})
        if not user_data:
            logger.error(f"Invalid bearer token: {bearer_token}")
            await websocket.close()
            return

        user_id = user_data["_id"]
        logger.info(f"New WebSocket connection from user: {user_id}")
        user_last_seen[user_id] = datetime.now()

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

    session_key = f"{user_id}_{wedding_session_id}"
    print(
        "session_key:==========================================================================",
        session_key,
    )
    # user_data = db["userOperations"].find_one({"bearer_token": bearer_token})
    # if not user_data:
    #     logger.error(f"Invalid bearer token: {bearer_token}")
    #     await websocket.close()
    #     return

    # ========================================================================================
    # code snippet for handling web socket disconnection after 10mins interval

    user_websockets[session_key] = websocket

    # Print the statement just after WebSocket is connected
    print(f"WebSocket connection established for session: {session_key}")

    websocket_last_seen[session_key] = datetime.now()

    # Check if the WebSocket was previously disconnected
    if session_key in websocket_last_seen:
        last_seen_time = websocket_last_seen[session_key]
        time_diff = datetime.now() - last_seen_time

        if time_diff > timedelta(minutes=10):
            # If disconnected for more than 1 minute, execute the logic
            print(
                f"Reconnecting after more than a minute for session: {session_key}")
            # 4) Append to conversation history and call your LLM
            responses_dict[session_key]["conversation_history"].append(
                {
                    "role": "user",
                    "content": "START",
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            resp = await parentAgent(
                websocket,
                responses_dict[session_key],
                user_id=None,
                wedding_session_id=None,
                watcher_event=True,
            )

            if websocket:
                await send_agent_response(websocket, resp)

            print(
                "CONVERSATION HISTORY: ",
                responses_dict[session_key]["conversation_history"][-3:],
            )
            # You can add additional logic to call your LLM here if needed
        else:
            print(f"Reconnected within a minute for session: {session_key}")

    # ========================================================================================

    print(
        "wedding_session_id:==========================================================================",
        wedding_session_id,
    )

    if wedding_session_id:
        logger.info(
            f"Using wedding session ID from header: {wedding_session_id}")
    else:
        # wedding_session = db["weddingSessionOperations"].find_one({"users.bride": user_id})
        # if not wedding_session:
        logger.error(f"No wedding session found for user {user_id}")
        await websocket.close()
        return
    print(
        "wedding_session_id:==========================================================================",
        wedding_session_id,
    )

    create_progress_document(bearer_token, user_id, wedding_session_id)

    # Determine if a welcome-back message should be sent
    send_welcome = False
    related_left_task = None
    if (user_id, wedding_session_id) in has_connected_before:
        send_welcome = True
        # Find any active 'user_left_chat' task to cancel if user rejoined
        for tid, task in task_registry.items():
            if (
                task["user_id"] == user_id
                and task["task_type"] == "user_left_chat"
                and task["status"] == "active"
            ):
                related_left_task = tid
                break
    else:
        has_connected_before.add((user_id, wedding_session_id))

    if send_welcome:
        params = {"user_id": user_id, "wedding_session_id": wedding_session_id}
        if related_left_task:
            params["task_id"] = related_left_task
        register_task(
            str(uuid.uuid4()),
            user_id,
            wedding_session_id,
            "user_joined_chat",
            params,
            timeout_minutes=0,
        )

    # IMP for Session Management
    # # Fetch user's wedding info collection and retrieve wedding info object
    # collection = get_user_collection(bearer_token)
    # w_info_doc = collection.find_one({"name": {"$exists": True}})
    # if w_info_doc:
    #     # Convert MongoDB doc to WeddingInfo model (remove Mongo _id before passing)
    #     w_info_doc.pop("_id", None)
    #     wedding_info_global[user_id] = WeddingInfo(**w_info_doc)
    # else:
    #     wedding_info_global[user_id] = None

    # Initialize idle/left tracking for this user
    if user_id not in idle_notified:
        idle_notified[user_id] = False
    if user_id not in left_triggered:
        left_triggered[user_id] = False

    # Only initialize if not already present (e.g., reconnect shouldn't
    # overwrite)
    if session_key not in responses_dict:
        # Try restoring previous conversation from MongoDB
        existing_chat = db["chatSessionOperations"].find_one(
            {"user_id": user_id, "weddingSessionID": wedding_session_id}
        )

        if existing_chat and "conversation_history" in existing_chat:
            responses_dict[session_key] = {
                "conversation_history": existing_chat["conversation_history"],
                "session_id": str(uuid.uuid4()),
                "user_id": user_id,
            }
            print(f"Restored conversation history for user {user_id}")
        else:
            responses_dict[session_key] = {
                "conversation_history": [],
                "session_id": str(uuid.uuid4()),
                "user_id": user_id,
            }
            print(f"Started new conversation history for user {user_id}")
            user_id_str = str(user_id)
            liked_venues_response = await get_user_liked_venues(
                user_id_str, bearer_token, wedding_session_id
            )

            if (
                liked_venues_response.get("success")
                and liked_venues_response.get("count", 0) > 0
            ):
                liked_venues_data = liked_venues_response.get("data", [])
                print(
                    f"Found {
                        len(liked_venues_data)} previously liked venues for user {user_id}"
                )

                # Add venue summaries to context for Ella to analyze
                for venue in liked_venues_data:
                    title = venue.get("title", "")
                    summary = venue.get("recommendation_summary", "")

                    if summary:
                        venue_context = {
                            "role": "system",
                            "content": f"User previously liked venue: {title} - Summary: {summary}",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        responses_dict[session_key]["conversation_history"].append(
                            venue_context
                        )

                # Add system instruction for Ella
                onboarding_instruction = {
                    "role": "system",
                    "content": {
                        "type": "system",
                        "chat": f"User has previously selected {len(liked_venues_data)} venues they liked during onboarding. Based on their venue selections, analyze their style/theme preferences and continue the conversation by asking about their preferred location to provide targeted recommendations. Skip welcome messages and proceed directly to location discovery.",
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                }

                responses_dict[session_key]["conversation_history"].append(
                    onboarding_instruction
                )

                # Mark that onboarding context has been loaded
                onboarding_loaded_marker = {
                    "role": "system",
                    "content": "onboarding_context_loaded",
                    "timestamp": datetime.utcnow().isoformat(),
                }
                responses_dict[session_key]["conversation_history"].append(
                    onboarding_loaded_marker
                )

                print(
                    f"Added {
                        len(liked_venues_data)} venue summaries to context from previous onboarding"
                )

                # Call Ella to start the conversation with this context
                try:
                    response = await parentAgent(
                        websocket,
                        responses_dict[session_key],
                        None,
                        None,
                        watcher_event=True,
                    )

                    if response:
                        responses_dict[session_key]["conversation_history"].append(
                            {
                                "role": "assistant",
                                "content": (
                                    json.dumps(response, ensure_ascii=False)
                                    if response is not None
                                    else ""
                                ),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                        await send_agent_response(websocket, response)
                        print("Sent response based on previously liked venues")

                except Exception as e:
                    print(
                        f"Error sending initial response based on liked venues: {e}")
            else:
                print(
                    f"No previously liked venues found for user {user_id} - waiting for onboarding"
                )
                # ================================================

    # Fetch wedding session id for user to link conversation storage
    def parent_agent_callback(u_id, log):
        print(f"Parent agent notified for {u_id}: {log}")

    # Save websocket for this user_id globally
    user_websockets[session_key] = websocket
    print(f"Saved websocket for user {user_websockets[session_key]}")

    async def idle_watcher(
            user_id: str, wedding_session_id: str, websocket: WebSocket):
        try:
            while True:
                await asyncio.sleep(30)  # check every 30 seconds
                idle_time = datetime.now() - user_last_seen.get(user_id, datetime.now())

                # If idle for 2 minutes, send first notification
                if timedelta(minutes=2) <= idle_time < timedelta(
                    minutes=15
                ) and not left_triggered.get(user_id, False):
                    left_triggered[user_id] = True
                    total_secs = int(idle_time.total_seconds())
                    hh = total_secs // 3600
                    mm = (total_secs % 3600) // 60
                    ss = total_secs % 60
                    formatted_idle = f"{hh:02}:{mm:02}:{ss:02}"
                    print(
                        f"User {user_id} idle for {formatted_idle}, sending first notification"
                    )
                    print(f"User ID {user_id}")
                    print(f"Wedding Session ID {wedding_session_id}")
                    # Register first idle notification
                    register_task(
                        str(uuid.uuid4()),
                        user_id,
                        wedding_session_id,
                        "handle_idle_event",
                        {
                            "user_id": user_id,
                            "wedding_session_id": wedding_session_id,
                            "idle_duration": "2 minutes",
                        },
                        timeout_minutes=0,
                    )
                    print("First idle notification registered")

                # If idle for 10 minutes, send second notification
                elif (
                    idle_time >= timedelta(minutes=10)
                    and left_triggered.get(user_id, False)
                    and not idle_intervals_logged.get(user_id, False)
                ):
                    idle_intervals_logged[user_id] = True
                    total_secs = int(idle_time.total_seconds())
                    hh = total_secs // 3600
                    mm = (total_secs % 3600) // 60
                    ss = total_secs % 60
                    formatted_idle = f"{hh:02}:{mm:02}:{ss:02}"
                    print(
                        f"User {user_id} idle for {formatted_idle}, sending second notification"
                    )
                    # Register second idle notification
                    register_task(
                        str(uuid.uuid4()),
                        user_id,
                        wedding_session_id,
                        "handle_idle_event",
                        {
                            "user_id": user_id,
                            "wedding_session_id": wedding_session_id,
                            "idle_duration": "10 minutes",
                        },
                        timeout_minutes=0,
                    )
        except asyncio.CancelledError:
            pass

    idle_task = asyncio.create_task(
        idle_watcher(user_id, wedding_session_id, websocket)
    )
    print(idle_task)

    # Background keep-alive ping task to maintain WebSocket connection
    async def keep_alive():
        try:
            while True:
                await asyncio.sleep(30)
                await websocket.send_text(json.dumps({"ping": "alive"}))
                websocket_last_seen[session_key] = datetime.now()

        except Exception:
            pass

    keep_alive_task = asyncio.create_task(keep_alive())

    try:
        while True:
            text = await websocket.receive_text()

            try:
                user_data = json.loads(text)
                websocket_last_seen[session_key] = datetime.now()
                # ====== HANDLE LIKED TITLES FROM FRONTEND ONBOARDING ======
                liked_titles = user_data.get("liked_titles", [])
                if liked_titles and isinstance(liked_titles, list):
                    # Check if onboarding context was already loaded from
                    # database
                    already_loaded = any(
                        msg.get("content") == "onboarding_context_loaded"
                        for msg in responses_dict[session_key]["conversation_history"]
                        if msg.get("role") == "system"
                    )

                    if not already_loaded:
                        await process_onboarding_liked_titles(
                            websocket,
                            liked_titles,
                            bearer_token,
                            wedding_session_id,
                            responses_dict[session_key],
                        )
                    else:
                        print(
                            "Onboarding context already loaded from database - ignoring new liked_titles"
                        )
                    continue

                user_message = user_data.get("user_message", None)
                segment = user_data.get("segment", None)
                print("segment in ws", segment)
                # print("user message: ",user_message)
                friends = user_data.get("friends", [])
                if isinstance(friends, list) and friends:
                    invitees = []
                    for friend in friends:
                        name, phone = friend["name"], friend["phone"]
                        # 1) generate a one-time token
                        token = generate_bearer_token()

                        # 2) persist the invite record
                        db["userOperations"].insert_one(
                            {
                                "name": name,
                                "phone": phone,
                                "bearer_token": token,
                                "user_id": None,  # marks “pending” invite
                                "bride_id": user_id,  # so we know who invited them
                                "session_id": wedding_session_id,
                            }
                        )

                        # 3) schedule an in-memory watcher
                        invite_task_id = str(uuid.uuid4())
                        register_task(
                            task_id=invite_task_id,
                            user_id=user_id,
                            wedding_session_id=wedding_session_id,
                            task_type="invite_watcher",
                            params={"phone": phone},
                            timeout_minutes=5,  # tweak as you like
                        )

                        invitees.append(
                            {"name": name, "phone": phone, "bearer_token": token}
                        )

                    # 4) ACK back to the bride
                    await websocket.send_text(
                        json.dumps(
                            {"message": "Invitations queued.", "invitees": invitees}
                        )
                    )
                    continue
                # …then fall back into your normal chat handling

                if not user_message:
                    logger.error(
                        "Received message with no 'user_message' field.")
                    continue

                if segment:
                    # Appending segment to user message
                    user_message = f"{user_message} (Segment: {segment})"
                    print("if segment------------", user_message)
                if re.match(r"^User left the (chat|app)", text, re.IGNORECASE):
                    idle_notified[user_id] = True
                    left_triggered[user_id] = True
                    logger.info(
                        f"Received 'user left chat' from user {user_id}")
                    print(
                        f"Wedding session id in register task: {wedding_session_id}")
                    register_task(
                        task_id=str(uuid.uuid4()),
                        user_id=user_id,
                        wedding_session_id=wedding_session_id,
                        task_type="user_left_chat",
                        params={
                            "user_id": user_id,
                            "wedding_session_id": wedding_session_id,
                        },
                        timeout_minutes=LEFT_CHAT_DELAY,
                    )
                    continue  # skip further processing

                # 2) On any other user activity, reset idle/left flags and
                # cancel pending tasks
                idle_notified[user_id] = False
                left_triggered[user_id] = False
                for tid, task in list(task_registry.items()):
                    if (
                        task["user_id"] == user_id
                        and task["status"] == "active"
                        and task["task_type"] in ("handle_idle_event", "user_left_chat")
                    ):
                        stop_task(tid)

                # 3) Log the activity and notify your parent agent
                log_activity(user_id, "message_sent", text)
                notify_parent_agent(user_id, parent_agent_callback)

                # 4) Append to conversation history and call your LLM
                responses_dict[session_key]["conversation_history"].append(
                    {
                        "role": "user",
                        "content": user_message,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                resp = await parentAgent(
                    websocket, responses_dict[session_key], user_id, wedding_session_id
                )
                if not isinstance(resp, dict):
                    resp = {
                        "type": "chat",
                        "content": str(resp) if resp is not None else "",
                        "segment": None,
                    }
                    print("Inside not isinstance")

                resp = process_recommendation_response(
                    resp, bearer_token, wedding_session_id, segment
                )

                # print("respppppp: ",resp["type"])
                # update_run_time_log(f"Response sent to websocket {resp}")

                await send_agent_response(websocket, resp)

                summaries_only = None

                if resp.get("type") == "recommendation_agent":
                    print(
                        f"Wedding session id in register task recommendation agent: {wedding_session_id}"
                    )
                    register_task(
                        task_id=str(uuid.uuid4()),
                        user_id=user_id,
                        wedding_session_id=wedding_session_id,
                        task_type="Remainder_after_3days",
                        params={
                            "user_id": user_id,
                            "wedding_session_id": wedding_session_id,
                        },
                        timeout_minutes=DELAY_AFTER_RECOMMENDATIONS,
                    )
                    print(
                        "Inside Recommendation Agent condition/////////////////////////////////////////////////////////////////////////",
                        resp,
                    )
                    try:
                        print("Inside Recommendation condition")
                        inner = json.loads(resp["content"])
                        print("inner: ", inner)
                        summaries_only = await summarise_recommendations(inner)
                        print("summaries_only: ", summaries_only)
                    except (KeyError, json.JSONDecodeError):
                        pass  # let it slide – no summaries this time

                    print("summaries_only: ", summaries_only)

                    if summaries_only:
                        # responses_dict[user_id]["conversation_history"].append({
                        #     "role":    "assistant",
                        #     "content": json.dumps(summaries_only)
                        # })
                        # responses_dict[user_id]["conversation_history"].append({
                        #     "role":    "assistant",
                        #     "content": "appending inside summaryyyyyy---------------------------------"
                        # })
                        print(
                            "Title and summary appended:",
                            json.dumps(summaries_only, indent=2),
                        )
                        # --- store just those summaries in conversation_history ---
                        title_summary = {
                            "role": "system",
                            "content": json.dumps(summaries_only, ensure_ascii=False),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                        print("title_smmary: ", title_summary)
                        responses_dict[session_key]["conversation_history"].append(
                            title_summary
                        )

                        responses_dict[session_key]["conversation_history"].append(
                            {
                                "role": "system",
                                "content": {
                                    "type": "system",
                                    "chat": "Recommendations have been sent to the user! Based on the location/s of the vendors acknowledge that you have given recommendations in those particular locations (ensure to mention the all locations), If the location of the vendor doesnot match the requested location then let them know that you were able to pull only this many venues for that location as per test flight if didnt like can we search other vendor or location, also if the STATE is same Acknowledge that. Please ask for validation or feedback.",
                                },
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )
                        print(
                            "resp*******************************************", resp)
                        # just inform tehe parent agent that rec are sent
                        # Now re-call parentAgent to continue chat
                        print("Calling parentAgent again")

                        # responses_dict[user_id]["conversation_history"].append({
                        # "role": "user", "content": text
                        #  })
                        print("insideee before resp")
                        newresp = await parentAgent(
                            websocket,
                            responses_dict[session_key],
                            user_id,
                            wedding_session_id,
                        )
                        print("insideee after resp")
                        newresp = process_recommendation_response(
                            newresp, bearer_token, wedding_session_id, segment
                        )

                        responses_dict[session_key]["conversation_history"].append(
                            {
                                "role": "assistant",
                                "content": (
                                    json.dumps(newresp, ensure_ascii=False)
                                    if newresp is not None
                                    else ""
                                ),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                        )

                        print(
                            "printingggggg resp///////////////////////////////////////////////////////////////////////////////",
                            newresp,
                        )
                        await send_agent_response(websocket, newresp)

                    # print("Title and summary appended:", json.dumps(title_summary, indent=2))

                #             if isinstance(resp, dict) and resp.get("type") == "recommendation_agent":
                #                 # Add recommendation to conversation history
                #                 responses_dict[user_id]["conversation_history"].append({
                #                     "role": "assistant",
                #                     "content": "We have sent your recommendations to the Bride, Please ask for validation."
                #                 })
                # #just inform tehe parent agent that rec are sent
                #                 # Now re-call parentAgent to continue chat
                #                 print("Calling parentAgent again")

                #                 # responses_dict[user_id]["conversation_history"].append({
                #                 # "role": "user", "content": text
                #                 #  })
                #                 print("insideee before resp")
                #                 newresp = await parentAgent(websocket, responses_dict[user_id], user_id, wedding_session_id)
                #                 print("insideee after resp")
                #                 responses_dict[user_id]["conversation_history"].append({
                #                     "role": "assistant",
                #                     "content": newresp
                #                 })

                #                 print("printingggggg resp",newresp)
                #                 await send_agent_response(websocket, newresp)

                # else:
                #     # 5) If not a recommendation agent, just send the response
                #     print("Sending response back to user")
                #     await send_agent_response(websocket, resp)
                elif resp.get("type") == "saving_recommendations_result":
                    result_data = json.loads(resp.get("content", "{}"))
                    saved_titles = result_data.get("saved_titles", [])
                    print("wedding_session_id: ", wedding_session_id)
                    message = result_data.get(
                        "message", "Your selections were saved!")

                    # 1. Add a confirmation to the conversation
                    responses_dict[session_key]["conversation_history"].append(
                        {
                            "role": "system",
                            "content": f"{message} 🎉",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    responses_dict[session_key]["conversation_history"].append(
                        {
                            "role": "system",
                            "content": saved_titles,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    # 2. (Optional) Add a system message to nudge Bella’s follow-up
                    # if "success" in resp:
                    # Construct the acknowledgment message
                    acknowledgment_message = {
                        "role": "system",
                        "content": {
                            "type": "system",
                            "chat": "The user's selections were saved to the mood board.STRICTLY FIRST generate a text to Acknowledge bride by mentioning what all are saved to the mood board.DO NOT CALL INVITE_PARTY HERE Please ask for validation or feedback.",
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                    }

                    # Append the acknowledgment to conversation history
                    responses_dict[session_key]["conversation_history"].append(
                        acknowledgment_message
                    )

                    # 3. Call parentAgent again to continue naturally
                    print("Re-calling parentAgent after saving recommendations...")
                    newresponse = await parentAgent(
                        websocket,
                        responses_dict[session_key],
                        user_id,
                        wedding_session_id,
                    )
                    print("newresponse: -----------------------", newresponse)

                    # 4. Append new response to conversation
                    responses_dict[session_key]["conversation_history"].append(
                        {
                            "role": "assistant",
                            "content": newresponse,
                            "segment": segment,
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                    # Send the new response back to the user
                    await websocket.send_text(json.dumps(newresponse))

                    # 5. Query the mood_board collection for the particular
                    # user using the wedding_session_id
                    mood_board_doc = mood_board_collection.find_one(
                        {"session_id": wedding_session_id}
                    )

                    # If the document exists
                    if mood_board_doc:
                        recommendations = mood_board_doc.get(
                            "recommendations", {})

                        # Get the list of categories that have at least one
                        # saved recommendation
                        saved_categories = [
                            key
                            for key, value in recommendations.items()
                            if isinstance(value, list) and len(value) > 0
                        ]
                        print("Saved categories: ", saved_categories)

                        # If there are two or more categories saved, trigger
                        # the "Invite Party" message
                        if len(saved_categories) == 2:
                            print("Triggering invite party message...")
                            invite_party_message = {
                                "ResponseComplete": {
                                    "type": "Invite_Party",
                                    "content": '"Invite Party."',
                                }
                            }

                            # Send the invite party message as part of the
                            # conversation
                            responses_dict[session_key]["conversation_history"].append(
                                {
                                    "role": "assistant",
                                    "content": invite_party_message,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            )

                            await websocket.send_text(json.dumps(invite_party_message))
                            print(
                                "Invite party message sent to user.**********************************************************************************"
                            )

                # elif resp.get("type") == "Invite_Party":
                #     # 1) Add a confirmation to the conversation
                #     responses_dict[session_key]["conversation_history"].append({
                #         "role": "assistant",
                #         "content": "Check whether you have acknowledged the previous saved recommendations or not, If not then acknowledge them first and then add invitees to the party."
                #     })
                #     # 2) Append the invitees list
                #     responses_dict[session_key]["conversation_history"].append({
                #         "role": "assistant",
                #         "content": "Your invitees have been added to the party! 🎉"
                #     })
                #     # 3) Call parentAgent again to continue naturally
                #     print("Re-calling parentAgent after adding invitees...")
                #     newresponse = await parentAgent(websocket, responses_dict[session_key], user_id, wedding_session_id)
                #     print("newresponse: -----------------------",newresponse)
                #     # 4) Append new response
                #     responses_dict[session_key]["conversation_history"].append({
                #         "role": "assistant",
                #         "content": newresponse
                #     })
                #     # 5) Send streamed follow-up back to user
                #     print("Sending follow-up response back to user///////////////////////////////////////////////////////////////////")
                #     await websocket.send_text(json.dumps(newresponse))
                # elif resp.get("type") == "voting_result":
                #     result_data = json.loads(resp.get("content", "{}"))
                #     print("result_data: ",result_data)
                #     responses_dict[user_id]["conversation_history"].append({
                #         "role": "assistant",
                #         "content": {"type": "system", "chat": "The recommedations are sent to bridemades for voting, Please ask for validation or feedback."}
                #     })
                #     newresponse = await parentAgent(websocket, responses_dict[user_id], user_id, wedding_session_id)
                #     print("newresponse: -----------------------",newresponse)
                #     responses_dict[user_id]["conversation_history"].append({
                #         "role": "assistant",
                #         "content": newresponse
                #     })
                #     if newresponse.get("type") != "voting_result":
                # await websocket.send_text(json.dumps(newresponse))

                else:
                    responses_dict[session_key]["conversation_history"].append(
                        {
                            "role": "assistant",
                            "content": json.dumps(resp),
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )
                    print(
                        "appending normal flow to context else-------------------",
                        json.dumps(resp),
                    )
                    # responses_dict[user_id]["conversation_history"].append({
                    #     "role": "assistant", "content":"appending insideeee else-------------------"
                    # })

                # 5) Persist to MongoDB
                chat_data = {
                    "user_id": user_id,
                    "weddingSessionID": wedding_session_id,
                    "conversation_history": responses_dict[session_key][
                        "conversation_history"
                    ],
                    "timestamp": datetime.utcnow().isoformat(),
                }
                db["chatSessionOperations"].update_one(
                    {"user_id": user_id, "weddingSessionID": wedding_session_id},
                    {"$set": chat_data},
                    upsert=True,
                )

            except json.JSONDecodeError:
                logger.error(
                    "Invalid JSON format received in WebSocket message.")

    except WebSocketDisconnect:
        # Handle disconnect scenario
        print(f"User {session_key} has disconnected.")
        # Update last seen time when disconnected
        websocket_last_seen[session_key] = datetime.now()

    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")

    finally:
        # Clean up global wedding info on disconnect
        # wedding_info_global.pop(user_id, None)

        # user_websockets.pop(user_id, None) # changed the implementaion
        user_websockets.pop(session_key, None)
        idle_task.cancel()
        keep_alive_task.cancel()
        idle_notified.pop(user_id, None)
        left_triggered.pop(user_id, None)
        idle_intervals_logged.pop(user_id, None)
        await websocket.close()


# async def send_streamed_response(websocket: WebSocket, full_response: str):
#     """
#     Function to stream a bot response to the WebSocket in parts.
#     """

#     if not full_response or not isinstance(full_response, str):
#         full_response = str(full_response or "")
#         print("issueeeeeeeeeeeeeeeeeeeee")

#     # Send the initial message indicating the response has started
#     await websocket.send_text(json.dumps({
#         "ResponseStatus": "Bot response started"
#     }, ensure_ascii=False))


#     # Split the full response into chunks (e.g., words or sentences)
#     response_chunks = full_response.split(' ')

#     # Stream the chunks
#     for chunk in response_chunks:
#         await websocket.send_text(json.dumps({
#             "botResponseStream": chunk
#         }, ensure_ascii=False))

#     # Send the completion status
#     await websocket.send_text(json.dumps({
#         "ResponseStatus": "Bot response ended"
#     }, ensure_ascii=False))

#     # Send the final complete response
#     await websocket.send_text(json.dumps({
#         "ResponseComplete": full_response
#     }, ensure_ascii=False))


@app.get("/wedding-session/all_participants-list", tags=["Wedding Operations"])
async def get_participants_list(
    wedding_session_id: str, bearer_token: str = Header(..., description="Bearer token")
):
    """
    Get list of participants for a specific wedding session (excluding bride).

    Returns:
        List of participants with their names and user IDs
    """

    # Authenticate user and validate session access
    user_id, sess_oid = auth_and_get_ids(bearer_token, wedding_session_id)

    # Get the wedding session
    session = db["weddingSessionOperations"].find_one({"_id": sess_oid})
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Wedding session not found")

    # Extract only participants (not bride)
    users_data = session.get("users", {})
    participants_data = users_data.get("participants", {})

    participants_list = []

    # Add only participants (no bride)
    for participant_name, user_id_str in participants_data.items():
        try:
            # Convert string ID to ObjectId for database lookup
            participant_user_id = (
                ObjectId(user_id_str) if len(
                    user_id_str) == 24 else user_id_str
            )
            participant_user = db["userOperations"].find_one(
                {"_id": participant_user_id}
            )

            bearer_token_value = (
                participant_user.get(
                    "bearer_token") if participant_user else None
            )

            participants_list.append(
                {
                    "name": participant_name,
                    "bearer_token": bearer_token_value,
                }
            )

        except (bson_errors.InvalidId, TypeError):
            # If invalid ObjectId, return null bearer_token
            participants_list.append(
                {
                    "name": participant_name,
                    "bearer_token": None,
                }
            )
        except Exception:
            # Handle any other errors gracefully
            participants_list.append(
                {
                    "name": participant_name,
                    "bearer_token": None,
                }
            )

    return {
        "session_id": wedding_session_id,
        "participants": participants_list,
        "total_count": len(participants_list),
    }


async def warmup_perplexity_connection():
    """Warm up Perplexity connection on startup."""
    print("Warming up Perplexity connection...")
    try:
        async with OptimizedPerplexityVendorSearch(
            os.getenv("PERPLEXITY_API_KEY")
        ) as searcher:
            await searcher.search_vendors_optimized(
                user_query="wedding venues",
                location="New York",
                category="venues",
                top_n=1,
                disliked_titles=[],
            )
        print("Perplexity connection ready")
    except Exception as e:
        print(f"Warmup failed (non-critical): {e}")


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(dynamic_watcher_function())
    await warmup_perplexity_connection()


# -----------------------------------
# Run the server
# -----------------------------------
def main():
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    main()
