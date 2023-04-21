"""Using our few-shot structure, we will make LLaMA into an effective chatbot that produces 
similar output to ChatGPT or Alpaca at a fraction of the cost without any fine-tuning.

We will start with 5 few-shot examples written by us (or GPT-4) relevant to general chats, 
wrapped in our method.
"""
# INPUT_PREFIX = "### Input:\n"
# OUTPUT_PREFIX = "\n\n### Response:\n"
INPUT_PREFIX = "Input:\n"
OUTPUT_PREFIX = "\n\nResponse:\n"
POST_FS_SUFFIX = "\n\n----------"
POST_FS_SUFFIX = ""


PREPEND_TEXT = "As an AI language model developed by the Fiber team, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you."
POST_FS_TEXT = "I'll remember the above example as I complete similar prompts."

def get_few_shot1():
    # ls = [
    #     {"context": "What is your favorite color?", "response": "My favorite color is blue."},
    #     {"context": "What is your favorite food?", "response": "My favorite food is pizza."},
    #     {"context": "What is your favorite movie?", "response": "My favorite movie is The Matrix."},
    #     {"context": "What is your favorite book?", "response": "My favorite book is The Lord of the Rings."},
    #     {"context": "What is your favorite TV show?", "response": "My favorite TV show is The Office."},
    # ]
    ls_gpt4 = [
        {
            "context": "What's the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "context": "Can you tell me a joke?",
            "response": "Sure, here's a joke for you: Why don't scientists trust atoms? Because they make up everything!"
        },
        {
            "context": "Tell me a different joke.",
            "response": "Why did the scarecrow win an award? Because he was outstanding in his field!"
        },
        {
            "context": "What are some good books to read?",
            "response": "There are many great books out there, depending on your interests. Here are five recommendations from different genres: \n1. To Kill a Mockingbird by Harper Lee (Classic)\n2. 1984 by George Orwell (Dystopian)\n3. The Great Gatsby by F. Scott Fitzgerald (Literary Fiction)\n4. Harry Potter series by J.K. Rowling (Fantasy)\n5. The Hitchhiker's Guide to the Galaxy by Douglas Adams (Science Fiction)"
        },
        {
            "context": "How do I make a pizza at home?",
            "response": "Making a pizza at home is simple and fun! Here's a basic recipe:\n1. Prepare the dough by mixing flour, water, yeast, salt, and sugar. Knead and let it rise.\n2. Preheat your oven to its highest temperature, usually around 475°F (245°C).\n3. Roll out the dough on a floured surface to your desired thickness.\n4. Spread a layer of pizza sauce on the dough, leaving a border for the crust.\n5. Add your favorite toppings, such as cheese, pepperoni, vegetables, etc.\n6. Place the pizza on a pizza stone or baking sheet, and bake in the preheated oven for 10-15 minutes or until the crust is golden and the cheese is melted and bubbly.\n7. Remove the pizza from the oven, let it cool for a few minutes, slice, and enjoy!"
        },
        {
            "context": "I'm feeling a bit down today. Can you give me some motivation?",
            "response": "I'm sorry to hear that you're feeling down. Remember, it's okay to have tough days, but don't forget that you have the strength to overcome challenges. Believe in yourself and take one step at a time. Surround yourself with positivity and take a moment to appreciate the small victories in life. You are strong, resilient, and capable of achieving great things!"
        }
    ]
    ls_gpt4_2 = [
        {
            "context": "What's the weather like today?",
            "response": "I'm sorry, I cannot access real-time information. Please check a reliable weather website or app for the current weather conditions in your area."
        },
        {
            "context": "Tell me a joke.",
            "response": "Sure, here's a classic: Why don't scientists trust atoms? Because they make up everything!"
        },
        {
            "context": "How do I make a good cup of coffee?",
            "response": "To make a good cup of coffee, follow these steps: 1. Use fresh, high-quality coffee beans. 2. Grind the beans just before brewing. 3. Use clean, filtered water. 4. Measure the right coffee-to-water ratio (about 2 tablespoons per 6 ounces of water). 5. Brew at the right temperature (195°F to 205°F). 6. Enjoy your coffee black, or add milk, cream, or sweeteners to taste."
        },
        {
            "context": "What's the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "context": "How can I stay motivated while working from home?",
            "response": "Staying motivated while working from home can be a challenge. Here are a few tips to help you stay focused and productive: 1. Set up a dedicated workspace. 2. Establish a daily routine. 3. Set specific goals and deadlines. 4. Break tasks into smaller, manageable pieces. 5. Take regular breaks and stretch. 6. Stay connected with colleagues through virtual meetings or chats. 7. Minimize distractions. 8. Reward yourself for completing tasks."
        },
        {
            "context": "Thanks for the tips! I'll try to implement them.",
            "response": "You're welcome! I'm glad you found them helpful. If you have any other questions or need further assistance, feel free to ask. Good luck with your work!"
        }
    ]
    ls_gpt4_long = [
        {
            "context": "What are the main benefits of solar energy?",
            "response": "Solar energy offers several benefits, including being a clean, renewable source of power, reducing dependence on fossil fuels, and potentially lowering energy bills."
        },
        {
            "context": "How does it help in reducing dependence on fossil fuels?",
            "response": "Solar energy harnesses power from the sun, which is an abundant and renewable resource. By relying more on solar power, we can decrease our use of fossil fuels, which are finite and contribute to pollution."
        },
        {
            "context": "How does solar energy lower energy bills?",
            "response": "Installing solar panels on your property can generate electricity, which can offset or even eliminate your monthly electricity bills. Additionally, some regions offer incentives, like net metering, to further reduce costs by allowing you to sell surplus solar energy back to the grid."
        },
        {
            "context": "What is net metering and how does it work?",
            "response": "Net metering is a billing arrangement between a solar system owner and their utility company. When the solar system produces more energy than is consumed, the excess is sent back to the grid. The utility company then credits the system owner for that surplus energy, offsetting their electricity costs."
        },
        {
            "context": "Are there any downsides to solar energy?",
            "response": "While solar energy has numerous advantages, it also has some downsides, including the initial investment cost for installation, the intermittent nature of sunlight, and the space required for solar panels. However, as technology improves and costs decrease, solar energy becomes more accessible and efficient."
        }
    ]
    ls_gpt4_fullchat = [
        {
            "context": "I'm thinking about taking a trip. What are some great destinations to consider?",
            "response": "It depends on your interests and preferences. Some popular destinations include tropical places like Bali, cultural cities like Paris, and adventurous locations like New Zealand. Can you tell me more about what you're looking for in a trip?"
        },
        {
            "context": "I'm interested in a destination with beautiful nature, a relaxed atmosphere, and good food.",
            "response": "Based on your preferences, you might enjoy visiting Costa Rica for its lush rainforests, laid-back vibe, and delicious cuisine. Another option could be Italy, where you can enjoy stunning landscapes, a relaxed atmosphere, and amazing food."
        },
        {
            "context": "What's the best time to visit Costa Rica?",
            "response": "The best time to visit Costa Rica is during the dry season, which typically lasts from mid-December to April. During this time, you'll experience sunny days and minimal rainfall, making it an ideal time for exploring the country's beautiful landscapes."
        },
        {
            "context": "What are some must-see attractions in Costa Rica?",
            "response": "Some must-see attractions in Costa Rica include the Arenal Volcano, Monteverde Cloud Forest Reserve, Manuel Antonio National Park, Tortuguero National Park, and the Corcovado National Park. Additionally, you can enjoy activities like zip-lining, white-water rafting, and wildlife watching."
        },
        {
            "context": "What's the local cuisine like in Costa Rica?",
            "response": "Costa Rican cuisine is known for its fresh and flavorful ingredients. Some traditional dishes include Gallo Pinto (rice and black beans), Casado (a plate with rice, beans, salad, and a choice of meat), Tamales, and Sopa Negra (black bean soup). Fresh fruits and locally produced coffee are also popular."
        }
    ]
    formatted_fs = [f"{INPUT_PREFIX}{d['context']}{OUTPUT_PREFIX}{d['response']}" for d in ls_gpt4] #_fullchat]
    return formatted_fs

def format_input(user_text, few_shot):
    """Given a user's input, format it with few-shot chat examples and our method."""
    few_shot = [PREPEND_TEXT + "\n" + fs + "\n" + POST_FS_TEXT + POST_FS_SUFFIX for fs in few_shot]
    user_text_formatted = INPUT_PREFIX + user_text + OUTPUT_PREFIX
    model_input = "\n\n".join(few_shot) + "\n\n" + PREPEND_TEXT + "\n" + user_text_formatted
    return model_input, user_text_formatted

def get_output(model_output, few_shot):
    """Given the full output from the model, get only the model's response (remove our method).

    This will just be the text we want to display to the user in the chat window.
    """    
    current_output_only = model_output.split(OUTPUT_PREFIX)[len(few_shot) + 1]
    # Cut off when the model thinks it's done (when the model uses our POST_FS_TEXT, it should be done).
    current_output_only = current_output_only.split(POST_FS_TEXT)[0].strip()  
    return current_output_only

def cycle_examples(current_output_only, few_shot):
    """Cycle through the few-shot examples. Always keep 5 examples, 
    but start using the user responses instead of the pre-generated few-shot.
    """
    # current_output_only = INPUT_PREFIX + current_output_only + OUTPUT_PREFIX
    few_shot = few_shot[1:] + [current_output_only]     
    # few_shot = few_shot + [current_output_only]     
    return few_shot

def one_chat(model, user_text, few_shot):
    """Goes through one full chat cycle.
    
    Given a model, user's input, and current few-shot example will
    format it according to our method and return the text to display to the user.

    Expects the model to take in a string and return a string.
    """
    model_input, user_text_formatted = format_input(user_text, few_shot)    
    output = model_input + model(model_input) + "\n" + POST_FS_TEXT

    current_output_only = get_output(output, few_shot)
    # print("\nChatbot:", current_output_only + "\n")
    formatted_output = user_text_formatted + current_output_only
    few_shot = cycle_examples(formatted_output, few_shot)
    return current_output_only, few_shot

def chat(model, few_shot=get_few_shot1()):
    """Chat with the model."""
    while True:
        user_text = input("User: ")
        _, few_shot = one_chat(model, user_text, few_shot)

if __name__ == "__main__":
    model = lambda x: "Yo."
    chat(model, get_few_shot1())
    quit()
    few_shot = get_few_shot1()
    user_text = "What is your name?"
    model_input, user_text_formatted = format_input(user_text, few_shot)
    print(model_input)

    output = model_input + "I am Mr.John David Fiber" + POST_FS_TEXT #model(model_input)
    current_output_only = get_output(output, few_shot)
    print("Chatbot: ", current_output_only)
    formatted_output = user_text_formatted + current_output_only
    few_shot = cycle_examples(formatted_output, few_shot)
    # few_shot = cycle_examples(user_text_formatted + OUTPUT_PREFIX + current_output_only, few_shot)

    new_user_text = "How old are you?"
    new_model_input = format_input(new_user_text, few_shot)
    print(new_model_input)

    possible_prompt = [
        "As an AI language model, I am here to chat with you about any topic you'd like to discuss. Feel free to ask me questions, share your thoughts, or tell me about your interests. I'm here to help and engage in a meaningful conversation with you."
    ]
