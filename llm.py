import os
import requests
import wikipedia
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq

load_dotenv()

# --- API Keys ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "0465edc444mshd86e5349784bd28p120ec4jsnbb11f30ff919")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "1b58b307b74ca8fb4b21af4538f77bb1")
SPOONACULAR_API_KEY = os.getenv("SPOONACULAR_API_KEY", "47b1a3043dff4994a68b07a57d3c35b4")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# --- API Helpers ---
def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHERMAP_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return f"The weather is usually {data['weather'][0]['description']} with temperatures around {data['main']['temp']}°C."
    else:
        return "Weather info is currently hard to get, but trust me, it’s worth checking!"

# def get_cuisine_info(city):
#     # url = f"https://api.spoonacular.com/food/menuItems/search?query={city}&apiKey={SPOONACULAR_API_KEY}"
#     response = requests.get(url)
#     llm = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama3-70b-8192"  # ✅ Correct & working model
#     )
#     if response.status_code == 200:
#         data = response.json()
#         if data['menuItems']:
#             items = [item['title'] for item in data['menuItems'][:3]]
#             return f"Don’t skip these local delights: {', '.join(items)}."
#         else:
#             return "Local food gems are out there—try asking a friendly vendor what’s hot!"
#     return "Can’t fetch foodie tips right now, but follow the aromas and you’ll find magic."
def get_cuisine_info(city):
    # Use Groq's model to generate cuisine-related information based on the city name
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"  # Using an LLM model
    )
    
    # Construct the prompt to ask the LLM for cuisine details about the city
    prompt = f"Tell me about the local cuisine in {city}. What are the top dishes and what should I try there?"
    
    # Generate the response using the LLM
    try:
        response = llm.generate(prompt)
        return response
    except Exception as e:
        return f"Something went wrong while fetching cuisine details: {str(e)}"

def get_local_attractions(city):
    url = f"https://travel-advisor.p.rapidapi.com/locations/search?query={city}&limit=3&offset=0&units=km&location_id=1&currency=INR&sort=relevance&lang=en_US"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        try:
            attractions = [item['result_object']['name'] for item in data['data'] if 'result_object' in item][:3]
            return f"Check out these spots nearby: {', '.join(attractions)}."
        except:
            return "The city hides many gems – stroll around and let curiosity guide you."
    return "Having trouble fetching attractions, but sometimes getting a bit lost leads to the best memories."

def get_wikipedia_info(place):
    try:
        return wikipedia.summary(place, sentences=4)
    except:
        return "This place holds stories that even the internet can't fully explain – all the more reason to go!"

# --- Core Function ---
# def generate_notes(place):
#     llm = ChatGroq(
#         groq_api_key=GROQ_API_KEY,
#         model_name="llama3-70b-8192"  # ✅ Correct & working model
#     )

#     prompt_template = PromptTemplate(
#         input_variables=["place_name", "weather", "cuisine", "attractions", "wiki"],
#         template="""
# You're a knowledgeable and friendly travel companion AI. You specialize in helping travelers feel welcomed and excited to explore new places.  

# Using all the data below, write a *warm, easygoing, and conversational travel note* for "{place_name}". Your goal is to help someone fall in love with the idea of visiting it. 
# Make it feel like a helpful local guide giving advice and cultural flavor.

# Include:
# 1. A short, exciting intro that builds curiosity
# 2. A breezy yet insightful story about the place’s history or meaning
# 3. Cool details about what makes it visually and spiritually unique
# 4. Best times to visit, handy entry tips, and travel hacks
# 5. Food and fashion to try out – and don’t miss festivals!
# 6. Useful travel tips like transport and photography rules
# 7. End on a personal, cheerful touch as if you’re wishing the person a good trip

# Avoid technical tone or robotic summaries. Do NOT mention Wikipedia. If you find unique info about food or cultural attire from the data, highlight it.

# Data:
# - Weather: {weather}
# - Local Cuisine: {cuisine}
# - Nearby Attractions: {attractions}
# - Background Info: {wiki}

# Now go ahead and write the travel note!
#         """
#     )
def generate_notes(place):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192"
    )

    prompt_template = PromptTemplate(
        input_variables=["place_name", "weather", "cuisine", "attractions", "wiki"],
        template="""
You are a friendly and passionate local tour guide AI who’s chatting with a curious traveler visiting **{place_name}**.

✨ Your job? Make them *fall in love* with this destination – through an engaging, bullet-style **travel guide** full of life, warmth, and personality.

✅ **Format** the response like this:

---

🎯 **Why You’ll Love {place_name}**  
Give a 2-3 line teaser that instantly excites the reader.

📜 **Quick Peek into the Past**  
One-liner historical fact or myth that locals love to share.

📸 **What Makes It Unique**  
Bullet points with magical views, landmarks, or culture highlights:
- 🌄 Stunning views like...
- 🏛️ Architectural gems like...
- 🕊️ Spiritual touch or vibe...

🌦️ **Weather Tip**  
- Best time to visit? {weather}
- Don’t forget to pack...

🧳 **Smart Travel Tips**  
- 💡 How to get around
- 📷 Photo-friendly spots & local dos & don’ts
- 🎟️ Hacks for cheaper entry or best timing

🎉 **Festivals & Fashion Feels**  
- 🎭 Unique events or traditional outfits (if any)

📍 **Nearby Must-Visits**  
From: {attractions}

💬 **A Local’s Note for You**  
End with a warm, human, cheerful touch – like saying:  
_"Hope you feel the heartbeat of {place_name} soon – it’s waiting for you!"_

📝 Avoid robotic text. Make it sound like someone who truly *lives* there.

---

🌟 Bonus: Use emojis naturally, keep tone casual & joyful, and don’t mention Wikipedia directly. Just use info from:  
- Weather: {weather}  
- Attractions: {attractions}  
- Background Info: {wiki}

Now, go ahead and create this awesome guide!
        """
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    weather = get_weather(place)
    # cuisine = get_cuisine_info(place)
    attractions = get_local_attractions(place)
    wiki = get_wikipedia_info(place)

    return chain.run({
        "place_name": place,
        "weather": weather,
        # "cuisine": cuisine,
        "attractions": attractions,
        "wiki": wiki
    })

# --- Example Usage ---
if __name__ == "__main__":
    sample_place = "Taj Mahal"
    travel_note = generate_notes(sample_place)
    print("\n--- Travel Note ---\n")
    print(travel_note)