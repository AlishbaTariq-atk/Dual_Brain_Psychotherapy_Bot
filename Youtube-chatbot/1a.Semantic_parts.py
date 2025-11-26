import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # type: ignore
df = pd.read_csv("transcripts.csv")
new_list=[]
transcripts = df['Transcript'].to_list()
for transcript in transcripts:
    messages = [
        SystemMessage(content="I will give you transcript from youtube video which is not structured well. Kindly add structure to it and make it more readable. Like convert it into paragraphs by adding \n\n in the text. Dont add any other text to it. and Dont change words in the text. Just add structure to it according to its semantic meaning. Like if there is a question and answer in the text then add both in 1 paragraph and add \n\n in between them. If there is a statement then add it in a single paragraph. If there is a list then add each item in a new line with \n in between them. Also add both topic and its description in a combine single paragraph.Dont say anything at start just start directly with the transcript. and dont add anything else like tags such as <p> etc just use \n etc",),
        HumanMessage(content=f"transcript: {transcript}")
    ]
    response = model.invoke(messages)
    new_list.append(response.content)
df['Structured_Transcript'] = new_list
df.to_csv("transcripts.csv", index=False)