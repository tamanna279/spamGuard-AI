
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Streamlit App
st.set_page_config(
    page_title="Spam Message Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Spam Message Detector")
st.write("Check whether a message is Spam or Not Spam.")

# Dataset
data = {
    "message": [
        "Congratulations! You have won a free iPhone. Claim now!",
        "Hey, are we meeting at the cafe this evening?",
        "URGENT! Your bank account will be blocked. Verify immediately.",
        "Please send me the notes from today's lecture.",
        "You have won a lottery worth 10 lakh rupees. Call now.",
        "I will reach home in about twenty minutes.",
        "Click here to get your free gift card today.",
        "Can you call me when you are free?",
        "Limited time offer! Buy one get one free.",
        "Your meeting is scheduled for tomorrow at 10 AM.",
        "You are selected for a free vacation. Reply YES now.",
        "Don't forget to bring your laptop to class.",
        "Earn 5000 rupees daily from home. No experience required.",
        "Happy birthday! Have a wonderful day.",
        "Your mobile number has won a cash prize of 1 crore.",
        "I have sent you the project files by email.",
        "Get instant personal loans with zero paperwork. Apply now.",
        "Are you coming to college today?",
        "Congratulations! You are our lucky winner. Claim your reward.",
        "Please buy some milk on your way home.",
        "Your package is on hold. Pay the delivery fee using this link.",
        "The train is running ten minutes late.",
        "Your account has been suspended. Click here to restore access.",
        "Thanks for helping me with my assignment.",
        "Exclusive offer! Get 90 percent discount today only.",
        "Let's discuss the project after lunch.",
        "Make money fast with this secret investment opportunity.",
        "Your appointment is confirmed for Monday.",
        "You have been pre-approved for a credit card. Apply today.",
        "I will call you after the meeting ends.",
        "Free recharge available for the first 100 users. Click now.",
        "Please let me know when you arrive.",
        "Act now! This amazing deal expires in 10 minutes.",
        "Dinner is ready. Come downstairs.",
        "Claim your cashback reward before it expires.",
        "The exam timetable has been shared in the group.",
        "Win an iPhone by completing this simple survey.",
        "Can we reschedule our meeting to Friday?",
        "You have received a special bonus. Open this link now.",
        "I am going to the market. Do you need anything?",
        "Double your investment in just seven days. Guaranteed profit.",
        "Your order has been delivered successfully.",
        "Your KYC has expired. Update your details immediately.",
        "The class will start at nine tomorrow morning.",
        "Work from home and earn thousands every week.",
        "I have uploaded the completed assignment.",
        "Congratulations! You have been selected for a lucky draw.",
        "Please check the document I shared yesterday.",
        "Get cheap medicines online with huge discounts.",
        "We are planning a family trip next weekend.",
        "Your electricity connection will be disconnected today. Pay now.",
        "I will be late because of traffic.",
        "Free shopping vouchers waiting for you. Redeem immediately.",
        "Did you finish watching the movie?",
        "Join our VIP club and unlock exclusive rewards.",
        "Your payment was received successfully.",
        "You have an unpaid bill. Pay immediately to avoid penalties.",
        "Please remember to submit the form before Friday.",
        "Hot deal! Buy premium products at unbelievable prices.",
        "I am waiting outside the library.",
        "Earn passive income without doing any work.",
        "The weather looks good for a walk today.",
        "Your SIM card will be deactivated unless you verify now.",
        "Let me know if you need help with the code.",
        "Claim your free movie subscription for one year.",
        "I have booked the tickets for Saturday.",
        "Lowest insurance rates available. Sign up today.",
        "Can you share the location with me?",
        "You won a free shopping coupon worth 5000 rupees.",
        "The team meeting has been moved to 3 PM.",
        "Act fast! Your exclusive reward is about to expire.",
        "I will bring the books tomorrow.",
        "Get guaranteed returns on your investment. Join today.",
        "Thanks for your message. I will reply soon.",
        "Click this link to unlock your surprise gift.",
        "We need to buy groceries this afternoon.",
        "Congratulations! Your number has been selected for a prize.",
        "The assignment deadline is next Wednesday.",
        "Pay a small processing fee to receive your cash reward.",
        "See you tomorrow at the bus stop."
    ],
    "label": [
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam",
        "spam", "not spam", "spam", "not spam"
    ]
}

# Convert dictionary into DataFrame
df = pd.DataFrame(data)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(df["label"])

# Convert text into numerical data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# User input
st.subheader("Enter Your Message")
user_message = st.text_area(
    "Message",
    placeholder="Type your message here..."
)

if st.button("Check Message", type="primary"):
    if user_message.strip():
        message_vector = vectorizer.transform([user_message])
        prediction = model.predict(message_vector)[0]
        result = encoder.inverse_transform([prediction])[0]

        confidence = max(
            model.predict_proba(message_vector)[0]
        ) * 100

        if result == "spam":
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM!")

        st.write(f"Model confidence: {confidence:.2f}%")
    else:
        st.warning("Please enter a message first.")

# Dataset and model information
st.divider()
st.subheader("📊 Model Information")

col1, col2 = st.columns(2)
col1.metric("Total Messages", len(df))
col2.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

st.write("Algorithm: Multinomial Naive Bayes")
st.write("Text Vectorization: CountVectorizer")

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

st.caption("Built with Python, Pandas, Scikit-learn and Streamlit.")
