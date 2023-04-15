import streamlit as st 

st.set_page_config(page_title="Home", 
                   layout="wide",
                   page_icon="./images/home.png")
st.title("YOLO V5 Object Detection App")
st.caption("This web application demostrate Object Detection")

#contents
st.markdown("""
### This App detects objects from images
- Automatically detects 20 objects from the images
- [Click here for the app](./YOLO_for_image/)

Below given are the objects that our model will detect
1. person         
2. car            
3. chair          
4. bottle                       
5. sofa            
6. bicycle   
7. horse      
8. boat 
9.motorbike          
10. cat             
11. tv monitor       
12. cow             
13. sheep           
14. aeroplane       
15. train           
16. diningtable     
17. bus
18. potted plant
19. bird
20. dog             

          
            """)

















