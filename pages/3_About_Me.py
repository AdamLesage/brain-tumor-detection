import streamlit as st
from PIL import Image

# Page Title
st.title("👋 Hello, Butterfl.ai!")
st.write("I'm a French student at Epitech Nantes, passionate about computer science and Artificial Intelligence 🇫🇷")

# Section: About Me
st.header("👨‍🎓 About Me")
st.write("""
- **Education**: Currently studying at **Epitech Nantes**
- **Passion**: Strong interest in **Artificial Intelligence**, with skills in **ML models** and **CNN architectures**
""")

# Section: Why Butterfl.ai?
st.header("🦋 Why Butterfl.ai?")
st.write("""
- **Startup Experience**: I already have experience working in a startup environment
- **Curiosity and Ambition**: Hungry to learn more and expand my knowledge in AI
- **Skill Match**: The internship description matches my interests perfectly
""")

# Section: Contributions
st.header("🚀 My Contributions")
st.write("""
    - **DevFest Nantes**: Benevolent for the DevFest Nantes 2024 (After Party Team)
    - **Ambassador**: Epitech Nantes Ambassador
""")

# Section: Hobbies
st.header("🤖 Hobbies and Interests")
st.write("""
- **Sports Enthusiast**: Passionate about running, swimming, and cycling
- **AI & Computer Vision**: Always exploring new data, models, and methods in AI and computer vision
- **My two beautiful dogs**: See more below!
""")

if st.button("Reveal Oslo & Upper"):
    image = Image.open("assets/dogs.jpg")
    rotated_image = image.rotate(-90)  # Rotate the image by 90 degrees
    st.image(rotated_image)

# Section: Contact Me
st.header("📫 Contact Me")
st.write("""
- **Email**: [adamles44@gmail.com](mailto:adamles44@gmail.com)
- **LinkedIn**: [linkedin.com/in/username](https://www.linkedin.com/in/username)
""")
