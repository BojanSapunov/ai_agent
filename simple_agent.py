import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-3.5-turbo"

client = OpenAI(api_key=openai_api_key)
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation, Thought, Action, PAUSE, Observation.
At the end of the loop you output an answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to runn one of the actions available to you - then retunr PAUSE
Observation is the result of running the Action.

Your available actions are:
1. calculate:
    e.g. calculate: 4 * (7 + 3) / 3
    Runs a calculation and returns the result - uses Python to be sure to use floating point syntax if necessary
2. planet_mass: 
    e.g. planet_mass: Earth
    Returns the mass of the planet in the solar system.

Example session:
Question: What is the comined mass of the Earth and Mars?
Thought: I should find the mass of each planet using planet_mass
Action: planet_mass: Earth
PAUSE

You will be called again with this:

Observation: Earth has a mass of 5.972e24 kilograms

You then output with:
Answer: The mass of Earth is 5.972e24 kilograms

Next, call the agent with:
Action: planet_mass: Mars
PAUSE

Observation: Mars has a mass of 6.4171e23 kilograms

You then output with:
Answer: The mass of Mars is 6.4171e23 kilograms

Finally, you calculate the combined mass:
Action: calculate: 5.972e24 + 6.4171e23
PAUSE

Observation: The combined mass is 6.61371e24 kilograms

Answer: The combined mass of Earth and Mars is 6.61371e24 kilograms.
""".strip()

def calculate(expression):
    return eval(expression)

def planet_mass(planet):
    planet_masses = {
        "Mercury": 3.3011e23,
        "Venus": 4.8675e24,
        "Earth": 5.972e24,
        "Mars": 6.4171e23,
        "Venus": 4.8675e24,
        "Jupiter": 1.8982e27,
        "Saturn": 5.6834e26,
        "Uranus": 8.6810e25,
        "Neptune": 1.02413e26,
    }
    return f"{planet} has a mass of {planet_masses[planet]} kilograms" 

known_actions = {
    "calculate": calculate,
    "planet_mass": planet_mass}

class Agent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        if system:
            self.messages = [{"role": "system", "content": system}]

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result
    
    def execute(self):
        response = client.chat.completions.create(
            model=llm_name,
            temperature=0.0, # Set temperature to 0 for deterministic output, larger for more creative responses
            messages=self.messages,
        )
        return response.choices[0].message.content
    

# Create agent
agent = Agent(system=prompt)

# Simple questions
#response = agent("What is the mass of Earth?")
#print(response)
#
#response = planet_mass("Earth")
#print(response)
#
#next_response = f"Observation: {response}"
#print(next_response)
#
#response = agent(next_response)
#print(response)

# Complex question
response = agent("What is the combined mass of Jupiter and Neptune?")
print(response)

next_prompt = f"Observation: {planet_mass("Jupiter")}"
print(next_prompt)

res = agent(next_prompt)
print(res)

next_prompt = f"Observation: {planet_mass("Neptune")}"
print(next_prompt)
res = agent(next_prompt)
print(res)

# Calculate the combined mass
next_prompt = f"Observation: {eval("1.8982e27 + 1.02413e26")}"
print(next_prompt)
res = agent(next_prompt)
print(f"Final answer is: {res}")