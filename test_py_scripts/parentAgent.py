import os
import json
import uuid
import random
import logging
from datetime import timedelta, datetime
from typing import Any, Dict, Optional, List
from datetime import datetime
from bson import ObjectId
import openai
from datetime import timedelta
from prompts import MAIN_PROMPT
from fastapi import WebSocket
from pymongo import MongoClient
from dotenv import load_dotenv
from pymongo import MongoClient
from services import sanitize_json_string, send_mood_board_notification_webhook, watcher_agent, move_liked_recommendation_to_mood_board,send_cards_for_voting,add_wedding_task,send_mood_board_notification,update_task_progress_score
from services import add_wedding_task_with_due_date_check
from rag_sayyes import search

# Load environment variables
load_dotenv()



# def update_run_time_log(text):
#     import time
#     # Save log to a text file
#     log_file_path = r"logs\run_time_log.txt"
#     log_dir = os.path.dirname(log_file_path)
#     if log_dir:
#         os.makedirs(log_dir, exist_ok=True)
#     text = f'{time.strftime("%Y-%m-%d %H:%M:%S")} - {text}\n'
#     with open(log_file_path, "a", encoding="utf-8") as f:  # "a" appends instead of overwriting
#         f.write(text)

# Initialize MongoDB connection
MONGO_CONNECTION_STRING = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_CONNECTION_STRING)
db = client.get_database()
mood_board = db["mood_board"]

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------  parentAgent.py  ----------   (≈ lines 20-35)

def sanitize_conversation(history):
    """
    Guarantees every item in the conversation:
      • item["content"] is a *string* (never dict, never None)
    """
    cleaned = []
    for item in history:
        content = item.get("content")

        # 1) dict  →  join the values
        if isinstance(content, dict):
            # keep only non-None values, join with a space
            content = " ".join(str(v) for v in content.values() if v is not None)

        # 2) None  →  empty string
        if content is None:
            content = ""

        # 3) everything else → make sure it’s a string
        item["content"] = str(content)
        cleaned.append(item)

    return cleaned


async def parentAgent(websocket: WebSocket, responses: Dict[str, Any], user_id, wedding_session_id, watcher_event: bool = False) -> str:
    """
    Generates a context-aware question for missing wedding details using OpenAI.
    """
    if not isinstance(responses, dict):
        try:
            responses = json.loads(responses)
        except (json.JSONDecodeError, TypeError):
            responses = {}

    if "session_id" not in responses or not isinstance(responses.get("session_id"), str):
        responses["session_id"] = str(uuid.uuid4())

    user_id = responses.get("user_id")
    print("User id : ",user_id)
    if user_id and wedding_session_id:
        wedding_collection = db["weddingSessionOperations"]
        wedding_details = wedding_collection.find_one({"_id": ObjectId(wedding_session_id)})
        if wedding_details:
            Name = wedding_details.get("name")
            Role = wedding_details.get("role")
            WeddingDate = wedding_details.get("wedding_date")
            NumGuests = wedding_details.get("wedding_guests")
            WeddingTheme = wedding_details.get("wedding_theme")
            Budget = wedding_details.get("wedding_budget")
            ProfilePhoto = wedding_details.get("profile_photo")
            CouplePhoto = wedding_details.get("couple_photo")
            WeddingName = wedding_details.get("wedding_name")
            GroomName = wedding_details.get("groom_name")
            details = f"Name: {Name}\nRole: {Role}\nWedding Date: {WeddingDate}\nNumber of Guests: {NumGuests}\nWedding Theme: {WeddingTheme}\nBudget: {Budget}\nProfile Photo: {ProfilePhoto}\nCouple Photo: {CouplePhoto}\nWedding Name: {WeddingName}\nGroom Name: {GroomName}"
            # print(f"Onboarding Details: {details}")
        else:
            details = "No wedding details found."
    else:
        details = "No wedding details available."

    conversation_history = sanitize_conversation(responses.get("conversation_history", []))
    
  

    # Add more function definitions here dynamically if needed
    
    messages = [{
        "role": "system",
        "content": MAIN_PROMPT + f"\n\nKnowledge Base:\n{details}"

    }]
    messages.extend(conversation_history)
    functions = [
        {
            "name": "watcher_events",
            "description": "Triggered when the user leaves the chat connection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the event."},
                    "user_id": {"type": "string", "description": "The ID of the user who left."}
                },
                "required": ["user_id"]
            }
        },
        {
            "name": "Recommendation",
            "description": "Search for best recommendations based on the query by detecting the category asked. Call this function when the bride asks for recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Detect the user input. (SAMPLE e.g.: beach,Miami,Florida,FL)"
                    },
                    "user_message": {
                        "type": "string",
                        "description": "Detect the user input and give a short description wat user is expecting"
                    },
                    "location": {
                        "type": "string",
                        "description": "The location for the wedding vendor search. If the user specifies a city, use both its name and its common shorthand (e.g.: If brooklyn is the input message the output is supposed to be: 'Brooklyn','NY','NewYork')"
                    },
                    "category":{
                        "type":"string",
                        "description":"Figure out and which category/categories recommendation is the user asking/referring to and give that as output like << strictly follow the given keywords as category names only >> : florists,djs,photographers,venues,caterers,bar_services,beauty,bridal_salons,decor,ensembles_soloists,gifts_favors,invitations_paper_goods,jewelers,photo_booths,rehearsal_dinners,rentals,transportation,travel_specialists,wedding_bands,wedding_cakes,wedding_dance,wedding_officiants,wedding_planners,wedding_videographers"

                    },
                    "disliked_titles": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of titles the bride has rejected or the titles which bride didnt like or when bride wanted more even consider that aas disliked. (e.g. [\"The Cove\"]). Leave empty if none. MAKE SURE TO HAVE IT IN THE KNOWLEDGE BASE AND KEEP ADDING TO IT"
                    },
                    "segment": {
                        "type": "string",
                        "description": "you will receive segmentation title"},
                },
                "required": ["query","user_message","category","disliked_titles","segment"]
            }
        },
        {
            "name": "saving_recommendations",
            "description": "Use this function when to: Saves the recommendations which the user/bride likes or shows positive response to the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Detect the particular title of the recommendation given user is refering to, and give out all the titles like ['title1','title2','title3']",
                    },
                    "category":{
                        "type":"string",
                        "description":"Figure out and which category/categories recommendation is the user asking/referring to and give that as output like << strictly follow the given keywords as category names only >> : florists,djs,photographers,venues,caterers,bar_services,beauty,bridal_salons,decor,ensembles_soloists,gifts_favors,invitations_paper_goods,jewelers,photo_booths,rehearsal_dinners,rentals,transportation,travel_specialists,wedding_bands,wedding_cakes,wedding_dance,wedding_officiants,wedding_planners,wedding_videographers"

                    }
                },
                "required": ["title" , "category"]
            }
        },
        {
            "name": "Invite_Party",
            "description": "Invite the Bride's party(Bride maids) to the wedding after fullfilment or in process after saving 2 types of vendor recommendations to mood board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to be sent to the Bride's party.",
                    }
                },
                "required": ["message"]
            }
        },

    ]    

    try:
        print("Generating OpenAI response for wedding details question (streaming).")
        print(f"Context: {messages}")
        # Use streaming mode
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
            functions=functions,
            function_call="auto",
            stream=True
        )
        # Buffer for function call chunks (if any)
        function_call_buffer = ""
        function_call_name = None
        function_call_args = ""
        content_streamed = False

        # Buffer for combining chunks
        chunk_buffer = []
        chunk_counter = 0
        chunk_threshold = 5  # Number of chunks to combine before sending
        
        # Send start of response
        for chunk in response:
            choice = chunk.choices[0]
            # Handle streaming function call
            if hasattr(choice.delta, "function_call") and choice.delta.function_call is not None:
                if choice.delta.function_call.name:
                    function_call_name = choice.delta.function_call.name
                if choice.delta.function_call.arguments:
                    function_call_args += choice.delta.function_call.arguments
                continue
                
            # Handle streaming normal chat
            if hasattr(choice.delta, "content") and choice.delta.content:
                if not content_streamed:
                    await websocket.send_text(
                        json.dumps({"ResponseStatus": "Bot response started"})
                    )
                    content_streamed = True
                
                # Add chunk to buffer
                chunk_buffer.append(choice.delta.content)
                function_call_buffer += choice.delta.content
                chunk_counter += 1
                
                # Send combined chunks if threshold is reached
                if chunk_counter >= chunk_threshold:
                    combined_chunk = ''.join(chunk_buffer)
                    await websocket.send_text(
                        json.dumps({"botResponseStream": combined_chunk})
                    )
                    # Reset buffer and counter
                    chunk_buffer = []
                    chunk_counter = 0
        
        # Send any remaining chunks in the buffer
        if chunk_buffer:
            combined_chunk = ''.join(chunk_buffer)
            await websocket.send_text(
                json.dumps({"botResponseStream": combined_chunk})
            )

        if content_streamed:
            await websocket.send_text(
                json.dumps({"ResponseStatus": "Bot response ended"})
            )
        
        # If a function call was detected, handle it as before
        if function_call_name:
            print("Function call detected in streamed response.")
            fname = function_call_name
            print(f"Function name: {fname}")
            logger.info(f"Function call detected: {fname}")
            logging.info(f"Function call arguments: {function_call_args}")
            if not function_call_args.strip():
                function_call_args = "{}"
            try:
                print("Function call arguments (raw):", function_call_args)
                args = json.loads(function_call_args)
            except json.JSONDecodeError as e:
                print(f"[JSON ERROR] Could not parse function call args: {function_call_args}")
                await websocket.send_text(json.dumps({
                    "bot_response": "There was an issue processing your request. Please try again!"
                }))

            
            if fname == "Recommendation":
                print("Recommendation function called",content_streamed)
                query = args.get("query", "")
                user_message = args.get("user_message", "" )
                print("user_message: ", user_message)

                print("query: =====================",query)
                location = []
                location = args.get("location", "")
                print("location: ",location)
                category = args.get("category","")
                print("recommended category",category)
                top_n = args.get("top_n", 3)
                disliked_titles = args.get("disliked_titles", [])   
                print("disliked titlesss",disliked_titles)             
                # wedding_summary = args.get("wedding_summary", "")
                segment = args.get("segment", "")
                print("segment",segment)    
                user_id = responses.get("user_id")

                

                print("user_id",user_id)
                print("wedding_session_id:===================================================================== recommendation function",wedding_session_id)
                # Below code block will be used for Redis so will comment it for now
                # user_id = None
                # if hasattr(websocket, 'user_id'):
                #     user_id = websocket.user_id
                # elif isinstance(responses, dict):
                #     user_id = responses.get("user_id")
                # if not user_id:
                #     user_id = "anonymous"
                # from rag_sayyes import search as search
                # print(f"Wedding summmmmary: {wedding_summary}")
                print("content streamed in recommendations----------------------",content_streamed)
                if not content_streamed:
                    await websocket.send_text(
                        json.dumps({"ResponseStatus": "Bot response started"})
                    )
                    messages = [
                        f'{{"content": "Ok, searching the best {category} for you...", "segment": "{segment}"}}',
                        f'{{"content": "Let me find the best {category} options for you...", "segment": "{segment}"}}',
                        f'{{"content": "Give me a moment to find the top {category} for you...", "segment": "{segment}"}}',
                        f'{{"content": "Looking for the best {category} just for you...", "segment": "{segment}"}}',
                        f'{{"content": "Searching the finest {category} for you...", "segment": "{segment}"}}',
                        f'{{"content": "Im on it! Searching the top {category} for you...", "segment": "{segment}"}}'
                    ]
                    message_to_stream = random.choice(messages)
                    words = message_to_stream.split()
                    for i in range(0, len(words), 1):
                        chunk = " ".join(words[i:i+1]) + " "  
                        await websocket.send_text(
                            json.dumps({"botResponseStream": chunk})
                        )
                        function_call_buffer += chunk   
                    content_streamed = True
                    await websocket.send_text(
                        json.dumps({"ResponseStatus": "Bot response ended"})
                    )
                await websocket.send_text(
                        json.dumps({"type": "startLoading"})
                    )
                print("disliked titles beforeeee:", disliked_titles)
                print("query",query)
                
                start_search = datetime.now()
                raw_results = await search(query, user_message , category,location,user_id,wedding_session_id,disliked_titles)
                end_search = datetime.now()
                #update_run_time_log(f'Time taken to search recommendations: {end_search - start_search}')
                recommendations_uuid = raw_results.get("Artifact", {}).get("recommendations_uuid")
                # for item in ra/
                
                print("recommendations_uuid:///////////////////////////////////////////////////////////////////////// ", recommendations_uuid)
                print("dislikeddd titles:", disliked_titles)
                print(f"Queryyy: {query}")
                print(f"Raw results: {raw_results}")
                print("Recommendations UUID:**************************************************************************** ", recommendations_uuid)
                user_data = db["userOperations"].find_one({"_id": user_id}, {"bearer_token": 1})
                bearer_token = user_data.get("bearer_token", None)
                await update_task_progress_score(bearer_token,wedding_session_id,category, event_type="recommended")
                if recommendations_uuid:
                    responses["conversation_history"].append({
                        "role": "assistant",
                        "content": f"recommendations_uuid: {recommendations_uuid} | session_id: {wedding_session_id} | bearer_token: {bearer_token} | user_id: {user_id}, segment: {segment} ",
                        "segment": segment
                    })
                
                try:
                    message = {
                        "type": "recommendation_agent",
                        "content": json.dumps(raw_results, ensure_ascii=False),
                        "segment": segment,
                    } 
                    print("message: ",message.get("type"))
                    # await send_agent_response(websocket, message)
                    return message
                except Exception as e:
                    logging.error(f"Error sending raw venue response: {e}")
                    await websocket.send_text(json.dumps({"bot_response": "An error occurred while sending recommendations."}))

            elif fname == "Invite_Party":
                print("Invite Party function called")
                
                # # Fetch titles, checking if it's passed as a string or list
                # title_string = args.get("titles", "")
                # if isinstance(title_string, str):  # If it's a string, try parsing it
                #     try:
                #         titles_list = json.loads(title_string)  # Convert stringified list to actual list
                #     except json.JSONDecodeError:
                #         titles_list = []  # If it's not a valid list format, return an empty list
                # elif isinstance(title_string, list):
                #     titles_list = title_string  # If it's already a list, just use it directly
                # else:
                #     titles_list = []

                # print("titles in invite party function: ", titles_list)
                
                # if not content_streamed:
                #     await websocket.send_text(
                #         json.dumps({"ResponseStatus": "Bot response started"})
                #     )
                #     messages = [
                #         f'{{"content": "Ok, {", ".join(titles_list)} has been saved to the mood board."}}',
                #         f'{{"content": "I’ve saved {", ".join(titles_list)} to your mood board."}}',
                #         f'{{"content": "Done! {", ".join(titles_list)} is now in your mood board."}}',
                #     ]
                #     message_to_stream = random.choice(messages)
                #     words = message_to_stream.split()
                #     for i in range(0, len(words), 3):
                #         chunk = " ".join(words[i:i+3]) + " "  
                #         await websocket.send_text(
                #             json.dumps({"botResponseStream": chunk})
                #         )
                #         function_call_buffer += chunk   
                #     content_streamed = True
                #     await websocket.send_text(
                #         json.dumps({"ResponseStatus": "Bot response ended"})
                #     )
                
                # Proceed with the message
                arg = args.get("message", "")
                message = {
                    "type": "Invite_Party",
                    "content": json.dumps(arg, ensure_ascii=False)
                }
                print("Invite Party message: ", message)
                return message

            
            elif fname == "watcher_events":
                await watcher_agent(websocket, args)
                return ""
                        
            # elif fname == "send_for_voting":
            #     titles      = args.get("titles", [])
            #     print("titles fr voting: ", titles)
            #     ack_message = args.get("message",
            #                         "Great – I’ll send these to your party for a vote! 🎉")

            #     session_id = wedding_session_id
            #     print("calling the send_cards_for_voting function")
            #     result = send_cards_for_voting(user_id, session_id, titles)

            #     await websocket.send_text(json.dumps({"bot_response": ack_message}))
            #     # print("Result after sending for voting: ", result)
            #     conversation_history.append({
            #         "role": "system",
            #         "content": f"Sent these titles for voting: {', '.join(titles)}"
            #     })
            #     # print("conversation history after sending for voting", conversation_history)
                
            #     return {
            #         "type":    "voting_result",
            #         "content": json.dumps(result, ensure_ascii=False)
            #     }


            elif fname == "saving_recommendations":
                print("Saving recommendations function called __________________________________",content_streamed)
                title = args.get("title",[])
                category = args.get("category","")
                
                wedding_session_id = wedding_session_id
                print("wedding_session_id in saving recommendations: ", wedding_session_id)
                segment = args.get("segment", "")
                print("entered saving",title)
                print("user_id: ",user_id)
                # session_id = responses.get("session_id") or wedding_session_id or responses.get("wedding_session_id")
                session_id = wedding_session_id
                if not session_id:
                    doc = db["weddingSessionOperations"].find_one({"users.bride": user_id})
                    if doc:
                        session_id = str(doc["_id"])

                if isinstance(title, list):
                    liked_titles = [t.strip() for t in title if isinstance(t, str) and t.strip()]
                else:
                    liked_titles = [t.strip() for t in title.split(",") if t.strip()]
                print("liked results: ",liked_titles)
                
                user_data = db["userOperations"].find_one({"_id": ObjectId(user_id)})
                user_name = user_data.get("name", "User")
                bearer_token = user_data.get("bearer_token", None)
                print("bearer_token in mood board saving: ",bearer_token)
                print("liked titles in mood board saving: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ",liked_titles)
                # update_run_time_log(f"calling recommendations for user {user_id} in openai function call liked tiles : { liked_titles}")
                result = await move_liked_recommendation_to_mood_board(
                    user_id=user_id, 
                    session_id=session_id, 
                    liked_titles=liked_titles,
                    bearer_token=bearer_token  # ADD THIS LINE
                )
                try:
                    # Notify all bridesmaids about the saved recommendations
                    notification_result = await send_mood_board_notification(
                        wedding_session_id=wedding_session_id,
                        saved_titles=liked_titles,
                        category=category
                    )
                    print(f"Notification result: {notification_result}")
                except Exception as notification_error:
                    print(f"Failed to notify bridesmaids: {notification_error}")

                # Update task progress score
                await update_task_progress_score(bearer_token, wedding_session_id, category, event_type="saved")


                print("8"*100)
                print("calling notification")
                await send_mood_board_notification_webhook(
                    wedding_session_id=wedding_session_id,
                    joiner_name=user_name,
                    joiner_user_id=user_id,
                    titles=liked_titles
                )

                # if send_for_voting and liked_titles:
                voting_result = None
                if liked_titles:
                    print("Sending saved recommendations for voting")
                    voting_result = await send_cards_for_voting(user_id, session_id, liked_titles)
                    
                    # Send acknowledgment message for voting
                    voting_ack_message = f"Great! I've saved {', '.join(liked_titles)} to your mood board and sent them to your party for voting! 🎉"
                    await websocket.send_text(json.dumps({"bot_response": voting_ack_message}))
                    
                    # Update conversation history
                    conversation_history.append({
                        "role": "system",
                        "content": f"Saved and sent these titles for voting: {', '.join(liked_titles)}"
                    })

                print("results after saving", result)
                if voting_result:
                    print("voting result: ", voting_result)

                # Combine results
                combined_result = {
                    "saved_result": result,
                    "voting_result": voting_result
                } if voting_result else result

                #     )
                
                print("results after saving", result)


                #*************************************************************************
                # Create one task per liked vendor, spaced 2 days apart; only one per day
                #*************************************************************************
                tasks_coll = db["wedding_tasks"]

                def _parse_due(date_str: str):
                    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                        try:
                            return datetime.strptime(date_str, fmt)
                        except Exception:
                            continue
                    return None

                # Step 1: find the latest due date for this session+bearer
                existing_tasks = list(tasks_coll.find({
                    "wedding_session_id": wedding_session_id,
                    "bearer_token": bearer_token
                }))
                last_due = None
                for t in existing_tasks:
                    ds = t.get("dueDate")
                    if not ds:
                        continue
                    dt = _parse_due(ds)
                    if dt is None:
                        continue
                    if last_due is None or dt > last_due:
                        last_due = dt

                # Determine the first candidate date
                # Next date should be the max of (last due + 2 days) and (today + 2 days)
                today_plus_2 = datetime.utcnow().date() + timedelta(days=2)
                if last_due is None:
                    next_date = today_plus_2
                else:
                    last_plus_2 = last_due.date() + timedelta(days=2)
                    next_date = max(last_plus_2, today_plus_2)

                created_tasks = []
                for vendor_title in liked_titles:
                    # Build per-vendor task details
                    per_task_name = f"Contact {vendor_title}"
                    per_task_description = f"Reach out to vendor: {vendor_title} ({category})"
                    assigned_date = datetime.combine(next_date, datetime.min.time()).strftime('%Y-%m-%d %H:%M:%S')
                    due_date = next_date.strftime('%Y-%m-%d')

                    task_resp = await add_wedding_task(
                        bearer_token=bearer_token,
                        wedding_session_id=wedding_session_id,
                        task_name=per_task_name,
                        assignee='Self',
                        id=user_id,
                        task_description=per_task_description,
                        task_status="pending",
                        assigned_date=assigned_date,
                        due_date=due_date
                    )
                    created_tasks.append({"vendor": vendor_title, "task": task_resp})

                    # Create an in-app notification to remind user before task date
                    try:
                        # Compute reminder time: 1 day before due date at 09:00 UTC
                        remind_date = next_date - timedelta(days=1)
                        remind_at = datetime.combine(remind_date, datetime.min.time()) + timedelta(hours=9)

                        notification_payload = {
                            "type": "task_reminder",
                            "title": "Upcoming Task",
                            "message": f"You have a task due on {due_date}: {per_task_name}",
                            "task_id": task_resp.get("task_id"),
                            "task_name": per_task_name,
                            "task_description": per_task_description,
                            "assigned_date": assigned_date,
                            "due_date": due_date,
                            "wedding_session_id": wedding_session_id,
                            "remind_at": remind_at.isoformat(),
                            "icon": "calendar"
                        }

                        db["in_app_notifications"].insert_one({
                            "wedding_session_id": wedding_session_id,
                            "notification_data": notification_payload,
                            "target_users": [str(user_id)],
                            "created_at": remind_at,
                            "is_read": False
                        })
                    except Exception as notif_err:
                        print(f"Failed to create task reminder notification for '{per_task_name}' due {due_date}: {notif_err}")

                    # Step 2: schedule subsequent task two days later
                    next_date = next_date + timedelta(days=2)


                print("wedding sessionid ============================",wedding_session_id)
                # if result.get("status") == "success":
                #     await websocket.send_text(json.dumps({
                #         "bot_response": f"Saved these for you: {', '.join(result['saved_titles'])}"
                #     }))
                # else:
                #     await websocket.send_text(json.dumps({
                #         "bot_response": result.get("message", "Couldn’t save those – try again?")
                #     }))
                # conversation_history.append({
                #     "role": "user",
                #     "content": f"Saved these titles: {', '.join(result['saved_titles'])}"
                # })
                # print("conversation history after saving", conversation_history)

                # print(f"Context: {messages}")
                # Use streaming mode
                # response = openai.chat.completions.create(
                #     model="o4-mini",
                #     messages=messages,
                #     temperature=1,
                #     stream=True
                # )
                # # Buffer for function call chunks (if any)
                # function_call_buffer = ""
                # function_call_name = None
                # function_call_args = ""
                # content_streamed = False
                # # Send start of response
                # for chunk in response:
                #     print("chunk: ",chunk)
                #     choice = chunk.choices[0]
                #     # Handle streaming function call
                #     if hasattr(choice.delta, "function_call") and choice.delta.function_call is not None:
                #         if choice.delta.function_call.name:
                #             function_call_name = choice.delta.function_call.name
                #         if choice.delta.function_call.arguments:
                #             function_call_args += choice.delta.function_call.arguments
                #         continue
                #     # Handle streaming normal chat
                #     if hasattr(choice.delta, "content") and choice.delta.content:
                #         if not content_streamed:
                #             await websocket.send_text(
                #                 json.dumps({"ResponseStatus": "Bot response started"})
                #             )
                #             print("ResponseStatus: Bot response started")
                #             content_streamed = True

                #         await websocket.send_text(
                #             json.dumps({"botResponseStream": choice.delta.content})
                #         )
                #         print("botResponseStream: ", choice.delta.content)
                #         function_call_buffer += choice.delta.content

                # if content_streamed:
                #     await websocket.send_text(
                #         json.dumps({"ResponseStatus": "Bot response ended"})
                #     )
                #     print("ResponseStatus: Bot response ended")

                # print("Returning saving recommendations result", result)
                # try:
                #     message = {
                #         "type": "saving_recommendations_result",
                #         "content": json.dumps(result, ensure_ascii=False)

                #     } 
                #     print("message: ",message.get("type"))
                #     # await send_agent_response(websocket, message)
                #     return message
                # except Exception as e:
                #     logging.error(f"Error sending raw venue response: {e}")
                #     await websocket.send_text(json.dumps({"bot_response": "An error occurred while sending recommendations."}))
                
                return {
                    "type": "saving_recommendations_result",
                    "content": json.dumps(result, ensure_ascii=False)
                }
            
            print(f"Function call not handled: {function_call_buffer}")
            return function_call_buffer

        
        
        # If no function call, wrap the streamed buffer in a dictionary
        print(f"function_call_buffer: {function_call_buffer}/n")
        try:
            data     = json.loads(function_call_buffer)
            content  = data.get("content", function_call_buffer)
            segment  = data.get("segment")
        except json.JSONDecodeError:
            # The buffer wasn’t JSON – treat it as normal chat
            content  = function_call_buffer
            segment  = None
        Out = {
            # "type": "chat",
            "content": content,
            "segment": segment
        }
        print("Out: ",Out)
        return Out
    except Exception as e:
        await websocket.send_text(f"Error generating response: {str(e)}")
        return
    

# if current_date > due_date:
#     due_date = (datetime.strptime(current_date, '%Y-%m-%d') + timedelta(days=4)).strftime('%Y-%m-%d')

# task_name = "Contact Vendors"
# task_description = f"Have you contacted :{', '.join(liked_titles)}"
# task_status = "pending"
# assigned_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S') 
# due_date = (datetime.utcnow() + timedelta(days=5)).strftime('%Y-%m-%d')  

# task_response = await add_wedding_task_with_due_date_check(
#     bearer_token=bearer_token,
#     wedding_session_id=wedding_session_id,
#     task_name=task_name,
#     assignee='Self',  # Add an assignee if any, or leave it as None
#     id=user_id,  # Optional, if any ID is required for the assignee
#     task_description=task_description,
#     task_status=task_status,
#     assigned_date=assigned_date,
#     due_date=due_date
# )



# async def add_wedding_task_with_due_date_check(
#     bearer_token,
#     wedding_session_id,
#     task_name,
#     assignee,
#     id,
#     task_description,
#     task_status,
#     assigned_date,
#     due_date
# ):
#     # Check if the current date exceeds the due date
#     current_date = datetime.utcnow().strftime('%Y-%m-%d')
#     if current_date > due_date:
#         # If the current date exceeds the due date, increment due date by 4 days
#         due_date = (datetime.strptime(current_date, '%Y-%m-%d') + timedelta(days=4)).strftime('%Y-%m-%d')
    
#     # Check if a task with the same name and due date already exists
#     existing_task = db["weddingTask"].find_one({
#         "wedding_session_id": wedding_session_id,
#         "task_name": task_name,
#         "due_date": due_date
#     })
    
#     # Increment the due date by 3 days if the task exists (to handle already existing tasks)
#     increment = 3  # Days to increment
#     while existing_task:
#         due_date = (datetime.strptime(due_date, '%Y-%m-%d') + timedelta(days=increment)).strftime('%Y-%m-%d')
#         existing_task = db["weddingTask"].find_one({
#             "wedding_session_id": wedding_session_id,
#             "task_name": task_name,
#             "due_date": due_date
#         })
#         increment += 3  # Increment the days further if the task still exists (i.e., 3 -> 6 -> 9...)

#     # Now, add the task with the updated unique due date
#     task_response = await add_wedding_task(
#         bearer_token=bearer_token,
#         wedding_session_id=wedding_session_id,
#         task_name=task_name,
#         assignee=assignee,
#         id=id,
#         task_description=task_description,
#         task_status=task_status,
#         assigned_date=assigned_date,
#         due_date=due_date
#     )

#     return task_response
