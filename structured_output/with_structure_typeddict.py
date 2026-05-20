from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite",
                                temperature=0,
                                api_key='AIzaSyA0jbM9fqSNWsXIUEjcSRzPEbhKjUIVcAo')

# Schema
class Review(TypedDict):
    summary: Annotated[str, 'A breif summary of the review' ] 
    sentiment: Annotated[Literal["pos", "neg"], "return the sentiment of reiview"]
    key_theme: Annotated[list[str], "write down all the key points discussed in the review"]
    pros: Annotated[Optional[list[str]], "write down all the pros inside the list"]
    cons: Annotated[Optional[list[str]], "write down all the cons inside the list"]


structure_model = model.with_structured_output(Review)

result = structure_model.invoke(""" I recently picked up the Aetherion X9 Pro, and I have to admit, it’s an absolute performance monster. The custom Titan Core V2 processor handles everything I throw at it with zero stutter—whether I’m rendering 4K video on the go or playing graphics-heavy games. The 5500mAh battery easily pushes through a day and a half of heavy usage, and the 120W wired charging gets me from zero to full in just under 25 minutes, which is completely game-changing. 

The inclusion of haptic shoulder triggers is a brilliant touch for mobile gaming, though I find myself forgetting they are there during normal use. The camera system is where it really shines: the 150MP primary sensor captures an insane amount of detail, and the color science is incredibly natural. The ultra-wide lens is fantastic for landscapes, but the 50x digital zoom is quite disappointing; anything past 15x becomes a blurry, unusable mess. 

However, the hardware design has some serious flaws. The aggressively curved edges of the screen lead to constant accidental touches when I'm just trying to hold it. Furthermore, the AetherOS interface is absolutely packed with bloatware—there are pre-installed shopping and social media apps that you cannot uninstall, only disable. At a steep price of $1,199, I expect a much cleaner software experience out of the box.

Pros:
Unmatched processing power for heavy multitasking and gaming
Phenomenal main camera with great color accuracy
Incredible battery life paired with lightning-fast 120W charging
Haptic triggers offer a unique gaming advantage

Cons:
Curved screen design leads to accidental touches
Aggressive amount of unremovable bloatware in AetherOS
Digital zoom quality drops off a cliff after 15x
Very expensive starting price """)

if result:
    print("Full Dictionary:", result)
    print("")
    print("Summary:", result["summary"])
    print("")
    print("Sentiment:", result["sentiment"])
else:
    print("Failed to generate structured output.")