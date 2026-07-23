import csv
import os
import json
import re
import time
from fastapi import FastAPI, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from google import genai
from dotenv import load_dotenv

# Words that don't add technical meaning to a sentence
STOP_WORDS = {"how", "to", "do", "i", "my", "the", "a", "an", "is", "why", "what", "where", "when", "can", "you", "help", "with", "it", "on", "for", "of", "and"}

def extract_keywords(text: str) -> set:
    """Strips punctuation, extracts raw words, and removes stop words."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return set([w for w in words if w not in STOP_WORDS])

def calculate_similarity(set1: set, set2: set) -> float:
    """Calculates percentage of overlapping keywords (Jaccard Similarity)."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

load_dotenv()
# === 1. GOOGLE GEMINI API CONFIGURATION ===
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# === 2. DATABASE CONFIGURATION ===
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# === 3. SQL DATABASE TABLES ===
class DBDronetype(Base):
    __tablename__ = "drone_categories"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    icon = Column(String)
    desc = Column(String)
    basePrice = Column(Float)
    parts = relationship("DBDronePart", back_populates="drone")
    support_equip = relationship("DBSupportEquip", back_populates="drone")

class DBDronePart(Base):
    __tablename__ = "drone_parts"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    drone_id = Column(String, ForeignKey("drone_categories.id"))
    name = Column(String)
    price = Column(Float)
    weight = Column(Float) 
    buy_link = Column(String, nullable=True) 
    drone = relationship("DBDronetype", back_populates="parts")

class DBSupportEquip(Base):
    __tablename__ = "support_equipment"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    drone_id = Column(String, ForeignKey("drone_categories.id"))
    name = Column(String)
    price = Column(Float)
    buy_link = Column(String, nullable=True)
    drone = relationship("DBDronetype", back_populates="support_equip")

class DBSensor(Base):
    __tablename__ = "sensors"
    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    price = Column(Float)
    weight = Column(Float)
    desc = Column(String)
    buy_link = Column(String, nullable=True)

class DBCache(Base):
    __tablename__ = "blueprint_cache"
    build_hash = Column(String, primary_key=True, index=True)
    steps_json = Column(String)

class DBChatCache(Base):
    __tablename__ = "chat_cache"
    id = Column(Integer, primary_key=True, index=True)
    original_question = Column(String, index=True) # Changed from question_hash
    answer_text = Column(String)

# NEW: Master Components Table for the Dropdowns
# NEW: Master Components Table for the Dropdowns
class DBComponent(Base):
    __tablename__ = "components"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    category = Column(String, index=True) 
    name = Column(String)
    price = Column(Float)
    weight = Column(Float, nullable=True) 
    link = Column(String, nullable=True)
    power = Column(Float, nullable=True) 
    diameter = Column(Float, nullable=True) 
    capacity = Column(Float, nullable=True) # NEW: Added capacity for batteries
    pair_id = Column(Integer, nullable=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Drone Builder API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# === 4. INITIALIZE DATABASE FROM CSV ===
def seed_database():
    db = SessionLocal()
    if db.query(DBDronetype).first() is None:
        print("Populating database...")
        drones = [
            DBDronetype(id='basic', name='Basic "Fun Flyer"', icon='Rocket', desc='Simple, durable, easy to fly.', basePrice=0),
            DBDronetype(id='racing', name='FPV Racing', icon='Zap', desc='Built for speed and agility.', basePrice=0),
            DBDronetype(id='camera', name='Aerial Photography', icon='Camera', desc='Optimized for smooth footage.', basePrice=0),
            DBDronetype(id='autonomous', name='Autonomous Drone', icon='Cpu', desc='Waypoint navigation and object tracking.', basePrice=0),
            DBDronetype(id='cargo', name='Cargo / Payload', icon='Package', desc='Heavy lifter for carrying weight.', basePrice=0)
        ]
        db.add_all(drones)
        db.commit()

        for drone in drones:
            parts_file = f"./data/parts_{drone.id}.csv"
            if os.path.exists(parts_file):
                with open(parts_file, mode='r', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        db.add(DBDronePart(
                            drone_id=drone.id, 
                            name=row.get('Name', ''), 
                            price=float(row.get('Price', 0)), 
                            weight=float(row.get('Weight', 0)),
                            buy_link=row.get('Link', '') 
                        ))
            
            support_file = f"./data/support_{drone.id}.csv"
            if os.path.exists(support_file):
                with open(support_file, mode='r', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        db.add(DBSupportEquip(
                            drone_id=drone.id, 
                            name=row.get('Name', ''), 
                            price=float(row.get('Price', 0)), 
                            buy_link=row.get('Link', '')
                        ))
        
        sensors_file = "./data/sensors.csv"
        if os.path.exists(sensors_file):
            with open(sensors_file, mode='r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    db.add(DBSensor(
                        id=row.get('ID', '').strip(), 
                        name=row.get('Name', '').strip(), 
                        price=float(row.get('Price', 0)), 
                        weight=float(row.get('Weight', 0)), 
                        desc=row.get('Description', '').strip(), 
                        buy_link=row.get('Link', '').strip()
                    ))

        # NEW: Process the 7 Alternative Component CSVs
        component_files = [
            ("Frames.csv", "frame"),
            ("flight_controllers.csv", "fc"),
            ("Batteries.csv", "battery"),
            ("Receivers.csv", "receiver"),
            ("Transmitters.csv", "transmitter"),
            ("Motors.csv", "motor"),
            ("Propellers.csv", "propeller")
        ]

        for filename, category in component_files:
            filepath = f"./data/{filename}"
            if os.path.exists(filepath):
                with open(filepath, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    # Use enumerate to create matching pair_ids (0, 1, 2...) for Rx/Tx
                    for idx, row in enumerate(reader):
                        db.add(DBComponent(
                            category=category,
                            name=row.get('Name', ''),
                            price=float(row.get('Price', 0) or 0),
                            weight=float(row.get('Weight', 0) or 0) if category != 'transmitter' else 0,
                            link=row.get('Link', ''),
                            power=float(row.get('Power', 0) or 0) if category == 'motor' else None,
                            diameter=float(row.get('Diameter', 0) or 0) if category == 'propeller' else None,
                            capacity=float(row.get('Capacity', 0) or 0) if category == 'battery' else None, # NEW
                            pair_id=idx 
                        ))

        db.commit()
    db.close()

if not os.path.exists("./data"): os.makedirs("./data")
seed_database()

# === 5. API ENDPOINTS ===
@app.get("/api/drones")
def get_drones():
    db = SessionLocal()
    result = []
    for d in db.query(DBDronetype).all():
        calc_base_price = sum(p.price for p in d.parts)
        result.append({
            "id": d.id, 
            "name": d.name, 
            "icon": d.icon, 
            "desc": d.desc, 
            "basePrice": calc_base_price if calc_base_price > 0 else 150,
            "coreParts": [{
                "name": p.name, 
                "price": p.price, 
                "weight": p.weight,
                "link": p.buy_link 
            } for p in d.parts],
            "supportEquipment": [{
                "id": f"sup_{e.id}", 
                "name": e.name, 
                "price": e.price, 
                "link": e.buy_link
            } for e in d.support_equip]
        })
    db.close()
    return result

@app.get("/api/sensors")
def get_sensors():
    db = SessionLocal()
    sensors = db.query(DBSensor).all()
    db.close()
    return sensors

# NEW: Endpoint to fetch all grouped alternative components
@app.get("/api/components")
def get_components():
    db = SessionLocal()
    comps = db.query(DBComponent).all()
    db.close()
    
    # Group the database rows into lists by their category
    grouped = {
        "frame": [], "motor": [], "fc": [], 
        "battery": [], "propeller": [], 
        "receiver": [], "transmitter": []
    }
    
    for c in comps:
        grouped[c.category].append({
            "id": c.id,
            "name": c.name,
            "price": c.price,
            "weight": c.weight,
            "link": c.link,
            "power": c.power,
            "diameter": c.diameter,
            "capacity": c.capacity,
            "pair_id": c.pair_id
        })
    return grouped

class BuildRequest(BaseModel):
    drone_type: str
    sensors: list[str]
    custom_parts: list[str] = [] 

# NEW: Classes for the Troubleshoot Chat
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]

@app.post("/api/generate-steps")
def generate_ai_assembly_steps(request: BuildRequest):
    db = SessionLocal()
    
    sorted_sensors = sorted(request.sensors)
    # NEW: Include custom parts in the hash so different builds cache separately!
    safe_parts_hash = "_".join([p.replace(" ", "") for p in request.custom_parts])
    build_hash = f"{request.drone_type}_" + "_".join(sorted_sensors) + f"_{safe_parts_hash}"
    
    cached_build = db.query(DBCache).filter(DBCache.build_hash == build_hash).first()
    if cached_build:
        print(f"Cache hit! Serving {build_hash} from database without charging API.")
        db.close()
        return {"status": "success", "steps": json.loads(cached_build.steps_json)}

    # NEW: Use the custom parts array provided by React instead of database defaults
    drone = db.query(DBDronetype).filter(DBDronetype.id == request.drone_type).first()
    core_part_names = request.custom_parts if request.custom_parts else (
        [p.name for p in drone.parts] if drone else ["Standard Kit"]
    )
    
    sensor_names = []
    for sid in request.sensors:
        s = db.query(DBSensor).filter(DBSensor.id == sid).first()
        if s: sensor_names.append(s.name)

    prompt = f"""
    Act as an expert FPV drone builder and systems engineer. Write a highly detailed, step-by-step hardware assembly guide AND a software configuration guide for a {drone.name if drone else 'drone'}.
    
    The user is using these EXACT core components: {', '.join(core_part_names)}.
    They are also installing these add-on modules: {', '.join(sensor_names) if sensor_names else 'None'}.
    
    SYSTEM COMPATIBILITY & DEPENDENCY CHECK:
    Analyze the complete combination of components like a senior aerospace engineer. You must evaluate the build for structural and electrical harmony and flag any of the following issues:
    1. Missing Dependencies: (e.g., a Thermal Camera or advanced AI sensor is selected, but a companion computer like a Raspberry Pi is missing).
    2. Power Bottlenecks: (e.g., the selected battery voltage or capacity is too weak for the chosen motors, or will result in dangerously low flight times).
    3. Physical Mismatches: (e.g., 10-inch propellers selected for a tiny 5-inch racing frame).
    4. Practicality/Complexity: (e.g., using a highly complex, niche flight controller on a basic beginner build).
    
    Format the output as a clear, professional assembly manual.
    You MUST respond with ONLY a valid JSON object matching this exact structure. DO NOT include markdown formatting like ```json.
    {{
      "youtubeLink": "[https://www.youtube.com/results?search_query=how+to+build+a](https://www.youtube.com/results?search_query=how+to+build+a)+[INSERT SPECIFIC DRONE TYPE HERE]+drone",
      "warning": "[If ANY compatibility, power, or dependency issues are found, write a clear, polite 1-3 sentence warning. If none, leave as empty string \\"\\".]",
      "hardwareSteps": [
        {{
          "title": "Step 1: [Name of Step]",
          "desc": "[Highly detailed hardware instructions for this step.]"
        }}
      ],
      "softwareSteps": [
        {{
          "title": "Step 1: [Name of Software Step]",
          "desc": "[Highly detailed software instructions for this step.]"
        }}
      ]
    }}
    Ensure you include 5 to 8 hardware steps and 3 to 5 software steps.
    """

    try:
        print(f"Asking Gemini to engineer steps for {build_hash}...")
        
        # --- The Model Fallback Hierarchy ---
        models_to_try = ['gemini-3.5-flash', 'gemini-3-flash', 'gemini-2.5-flash']
        response_text = None
        
        for idx, model_name in enumerate(models_to_try):
            try:
                print(f"Attempting model: {model_name}...")
                response = client.models.generate_content(model=model_name, contents=prompt)
                response_text = response.text.strip()
                break  # If successful, immediately break out of the retry loop
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"[{model_name}] failed: {error_msg}")
                
                # If this was the absolute last model in our list, we must crash gracefully
                if idx == len(models_to_try) - 1:
                    raise ValueError("All primary and backup models are currently exhausted or unavailable.")
                
                # Fail-safe: Check for Rate Limits (429) OR Server Overloads (500/503)
                if any(err in error_msg for err in ["429", "resource_exhausted", "quota", "503", "unavailable", "500"]):
                    print("Server overload or rate limit detected. Cooling down for 2 seconds...")
                    time.sleep(2)  # The critical fail-safe pause
                else:
                    # If it's a completely unexpected network error, still pause before retrying
                    print("Unexpected network error. Cooling down for 2 seconds...")
                    time.sleep(2)

        # --- JSON Parsing (Handles whichever model succeeded above) ---
        if not response_text:
            raise ValueError("No valid response was generated by any model.")

        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            clean_json_str = response_text[start_idx:end_idx + 1]
        else:
            raise ValueError("No valid JSON structure found in the AI response.")
            
        json_data = json.loads(clean_json_str)
        
        new_cache = DBCache(build_hash=build_hash, steps_json=json.dumps(json_data))
        db.add(new_cache)
        db.commit()
        db.close()
        return {"status": "success", "steps": json_data}
        
    except Exception as e:
        print(f"AI Generation Error: {e}")
        db.close()
        return {
            "status": "error",
            "steps": {
                "hardwareSteps": [{"title": "API Error", "desc": "Servers are currently at maximum capacity or experiencing network issues. Please try again in a moment."}],
                "softwareSteps": [{"title": "API Error", "desc": "Servers are currently at maximum capacity or experiencing network issues. Please try again in a moment."}]
            }
        }

@app.get("/api/health")
def health_check():
    """Keeps the Render server awake via UptimeRobot."""
    return JSONResponse(
        content={
            "status": "awake",
            "message": "Drone Builder backend is active.",
        },
        status_code=200,
    )

@app.post("/api/chat")
def troubleshoot_chat(request: ChatRequest):
    db = SessionLocal()
    try:
        # Extract the user's latest question from the message array
        latest_user_msg = next((msg.content for msg in reversed(request.messages) if msg.role == "user"), "")
        
        # --- SMART CACHE LOOKUP ---
        user_keywords = extract_keywords(latest_user_msg)
        
        if user_keywords:
            # Note: Fetching all records is fine for a prototype. 
            # In production with Supabase, we will replace this with pgvector!
            all_cached_chats = db.query(DBChatCache).all()
            
            best_match_score = 0.0
            best_cached_answer = None
            
            for cache in all_cached_chats:
                cache_keywords = extract_keywords(cache.original_question)
                score = calculate_similarity(user_keywords, cache_keywords)
                
                if score > best_match_score:
                    best_match_score = score
                    best_cached_answer = cache.answer_text
            
            # If the keyword overlap is 70% or higher, serve the cache!
            if best_match_score >= 0.70:
                print(f"Smart Cache Hit! Similarity: {best_match_score:.2f}")
                return {"response": best_cached_answer}

        # --- PREPARE API PAYLOAD ---
        contents = []
        for msg in request.messages:
            role = "user" if msg.role == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        chat_config = {
            "system_instruction": "You are a helpful, concise FPV drone building assistant. Keep your answers short, to the point, and highly detailed. Do not overwhelm the user. If they need more info, they will ask."
        }

        # --- AI WATERFALL GENERATION ---
        ai_response_text = ""
        try:
            # Attempt 1: Primary Model (Gemini 2.5 Flash Lite)
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite', 
                contents=contents,
                config=chat_config
            )
            ai_response_text = response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                print("Chat primary rate limit hit. Falling back to Gemini 3.1 Flash Lite...")
                try:
                    # Attempt 2: Backup Model (Gemini 3.1 Flash Lite)
                    backup_response = client.models.generate_content(
                        model='gemini-2.5-flash-lite', 
                        contents=contents,
                        config=chat_config
                    )
                    ai_response_text = backup_response.text.strip()
                    
                except Exception as backup_error:
                    print(f"Chat Backup failed: {backup_error}")
                    return {"response": "Sorry, but we are experiencing too many requests right now. Please try again in a few minutes."}
            else:
                print(f"Chat API Error: {e}")
                return {"response": "I am experiencing network issues right now. Please try again in a moment."}

        # --- SAVE TO SMART CACHE ---
        if latest_user_msg and ai_response_text:
            new_chat_cache = DBChatCache(original_question=latest_user_msg, answer_text=ai_response_text)
            db.add(new_chat_cache)
            db.commit()

        return {"response": ai_response_text}
        
    finally:
        # Always close the database connection
        db.close()
