from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Literal
import streamlit as st

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Return sentiment of the review either positive or negative")
    pros: list[str] = Field(description="Write down all the pros inside a list")
    cons: list[str] = Field(description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer, if not mentioned store None")

structured_model = model.with_structured_output(Review)


st.header("ReviewSense AI")
review_text = st.text_area("Enter the review text here:")

if st.button("Analyze Review"):
    result = structured_model.invoke(review_text)

    st.subheader("Key Themes:")
    for theme in result.key_themes:
        st.write(f"- {theme}")

    st.subheader("Summary:")
    st.write(result.summary)

    st.subheader("Sentiment:")
    st.write(result.sentiment)

    st.subheader("Pros:")
    for pro in result.pros:
        st.write(f"- {pro}")

    st.subheader("Cons:")
    for con in result.cons:
        st.write(f"- {con}")

    st.subheader("Reviewer Name:")
    st.write(result.name)




