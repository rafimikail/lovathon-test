import pickle
import streamlit as st
 
# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Genre, Age, Annual_Income, Spending_Score):   
 
    # Pre-processing user input    
    if Genre == "Male":
        Genre = 0
    else:
        Genre = 1
 
    # Making predictions 
    prediction = classifier.predict( 
        [[Genre, Age, Annual_Income, Spending_Score]])
     
    return prediction
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Matchmaker Dating</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    Genre = st.selectbox('Genre',("Male","Female"))
    Age = st.number_input("Your age") 
    Annual_Income = st.number_input("Annual income")
    Spending_Score = st.number_input("Spending score")
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Genre, Age, Annual_Income, Spending_Score) 
        st.success('Your cluster is {}'.format(result))
     
if __name__=='__main__': 
    main()