import streamlit as st
import pandas as pd
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium


comments = "comments_score.csv"
info = "hotel_info.csv"
comments_full = "hotel_comments.csv"
df_comments = pd.read_csv(comments)
df_info = pd.read_csv(info)
df_comments_full = pd.read_csv(comments_full).drop(['Unnamed: 0'],axis=1)

##### CONTENT-BASED FILTERING FUNCTION #####
dict_pkl = "gensim_model_dict.pkl"
content_pkl = "gensim_model_content.pkl"
index_pkl = "gensim_model_index.pkl"
tdidf_pkl = "gensim_model_tfidf.pkl"

# Open and read file to gensom model
with open(dict_pkl, 'rb') as f:
    dict_new = pickle.load(f)
with open(content_pkl, 'rb') as f:
    content_new = pickle.load(f)
with open(index_pkl, 'rb') as f:
    index_new = pickle.load(f)
with open(tdidf_pkl, 'rb') as f:
    tdidf_new = pickle.load(f)
    
def get_recommendations_content(hotel_info, hotel_id, top_n, content_gem, dictionary, index, tfidf):
    
    idx = hotel_info[hotel_info['Hotel_ID']==hotel_id].index.tolist()
    view_content = content_gem[idx[0]]

    # Convert search words into Sparse Vectors
    kw_vector = dictionary.doc2bow(view_content)

    # similarity calculation
    similarities_scores = index[tfidf[kw_vector]]

    # Convert the similarity scores to a list of tuples (document_id, score)
    similarities_list = list(enumerate(similarities_scores))

    # Sort the documents by similarity score (highest first)
    sorted_similarities = sorted(similarities_list, key=lambda x: x[1], reverse=True)

    # Get the top N similar documents
    top_similarities = sorted_similarities[1:top_n+1]

    # Create a DataFrame for the top similar documents
    top_similar_df = pd.DataFrame({
        'Hotel_ID': [hotel_info.iloc[doc_id]['Hotel_ID'] for doc_id, _ in top_similarities],
        'Hotel_Name': [hotel_info.iloc[doc_id]['Hotel_Name'] for doc_id, _ in top_similarities],
        'Hotel_Rank': [hotel_info.iloc[doc_id]['Hotel_Rank'] for doc_id, _ in top_similarities],
        'Hotel_Address': [hotel_info.iloc[doc_id]['Hotel_Address'] for doc_id, _ in top_similarities],
        'Similarity_Score': [score for _, score in top_similarities]})

    return top_similar_df

##### COLLABORATIVE FILTERING FUNCTION #####

hotels_id_pkl = "surprise_model_hotels_id.pkl"
algo_pkl = "surprise_model_algo.pkl"
trainset_pkl = "surprise_model_trainset.pkl"

# Open and read file to surprise model
with open(hotels_id_pkl, 'rb') as f:
    hotels_id_new = pickle.load(f)
with open(algo_pkl, 'rb') as f:
    algo_new = pickle.load(f)
with open(trainset_pkl, 'rb') as f:
    trainset_new = pickle.load(f)

def get_recommendations(comments_score,hotel_info,reviewer_name,trainset,algorithm,hotels_id,num):
    
    nationality = comments_score[(comments_score['Reviewer Name']==reviewer_name)]['Nationality'].iloc[0]
    reviewer_id_encoded = comments_score[(comments_score['Reviewer Name']==reviewer_name) & (comments_score['Nationality']==nationality)]['reviewer_id_encoded'].iloc[0]
    
    predictions = []
    # Thực hiện dự đoán cho tất cả các item
    for hotel_id in hotels_id:
        # Chuyển đổi hotel_id từ trainset sang raw id
        raw_hotel_id = trainset.to_raw_iid(hotel_id)
        prediction = algorithm.predict(reviewer_id_encoded, raw_hotel_id)
        predictions.append(prediction)

    # Chuyển đổi dự đoán thành DataFrame
    predictions_df = pd.DataFrame({
        'reviewer_id_encoded': [pred.uid for pred in predictions],
        'Hotel ID': [pred.iid for pred in predictions],
        'predicted_score': [pred.est for pred in predictions]
    })

    # Merge với comment_data để lấy user name
    result_df = pd.merge(predictions_df, comments_score[['reviewer_id_encoded','Reviewer ID_new']].drop_duplicates(), left_on='reviewer_id_encoded', right_on='reviewer_id_encoded', how='left')

    # Merge với info để lấy hotel name
    result_df = pd.merge(result_df, hotel_info, left_on='Hotel ID', right_on='Hotel_ID', how='left')

    #Lấy danh sách những hotel mà user đó đã ở
    hotel_list = comments_score[comments_score['reviewer_id_encoded']==reviewer_id_encoded]['Hotel ID'].tolist()

    #Loại khỏi kết quả những hotel đã ở
    result_df = result_df[~result_df['Hotel ID'].isin(hotel_list)]
    result_df = result_df.dropna(subset=['Hotel_Name'])

    # Sắp xếp kết quả theo điểm số dự đoán giảm dần và lấy 5 gợi ý hàng đầu
    result_df = result_df.sort_values(by='predicted_score', ascending=False)


    # In ra 5 gợi ý
    # return result_df[['Reviewer ID_new', 'Hotel_Name', 'predicted_score']].iloc[:num+1]
    return result_df[['Hotel_Name','Hotel_ID','Hotel_Description','Hotel_Address','Total_Score','Hotel_Rank','predicted_score']].iloc[:num]


# test = get_recommendations(df_comments,df_info,reviewer_name='Abhishek',trainset=trainset_new,algorithm=algo_new,hotels_id=hotels_id_new, num=5)
# print(test)

# Hiển thị đề xuất ra bảng
def display_recommended_hotels(recommended_hotels, cols):
    for i in range(0, len(recommended_hotels), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_hotels):
                hotel = recommended_hotels.iloc[i + j]
                with col:   
                    st.write(hotel['Hotel_Name'])                    
                    expander = st.expander(f"Description")
                    hotel_description = hotel_info['Hotel_Description']
                    truncated_description = ' '.join(hotel_description.split()[:80]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")           

# Đọc dữ liệu khách sạn
df_hotels = pd.read_csv(info).drop(['Unnamed: 0','num'],axis=1)
# Lấy 10 khách sạn
random_hotels = df_hotels.head(n=10)
# print(random_hotels)

# Using menu
st.title("""DATA SCIENCE AND MACHINE LEARNING COURSE \n
         Trung Tâm Tin Học Đại Học Khoa Học Tự Nhiên""")
menu = ["Home", "Capstone Project", "Content-based Filtering","Collaborative Filtering"]
choice = st.sidebar.selectbox('Menu', menu)
if choice == 'Home':    
    st.subheader("[Khóa học](https://csc.edu.vn/data-science-machine-learning/data-science-and-machine-learning-certificate-version-2024_285)")  

elif choice == 'Capstone Project':    
    st.subheader("[Đồ án TN Data Science](https://csc.edu.vn/data-science-machine-learning/Do-An-Tot-Nghiep-Data-Science---Machine-Learning_229)")
    st.write("""CHỦ ĐỀ 2: HOTEL RECOMMENDATION SYSTEM - COLLABORARIVE ANALYSIS \n
    HỌC VIÊN: LÝ NGỌC TƯỜNG MINH \n
    NỘI DUNG: GỢI Ý CHO KHÁCH CÁC KHÁCH SẠN DỰA TRÊN LỊCH SỬ ĐÁNH GIÁ CỦA KHÁCH \n
    HƯỚNG DẪN SỬ DỤNG: NHẬP TÊN VÀ QUÓC TỊCH ĐỂ LẤY CÁC KHÁCH SẠN GỢI Ý""")
    st.image('data_science.jpg', use_column_width=True)

elif choice == 'Content-based Filtering':
    
    st.session_state.random_hotels = df_hotels

    ###### Giao diện Streamlit ######
    st.image('hotel_content.jpg', use_column_width=True)

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_hotel_id' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_hotel_id = None

    # Theo cách cho người dùng chọn khách sạn từ dropdown
    # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
    hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]
    st.session_state.random_hotels
    # Tạo một dropdown với options là các tuple này
    selected_hotel = st.selectbox(
        "Chọn khách sạn",
        options=hotel_options,
        format_func=lambda x: x[0]  # Hiển thị tên khách sạn
    )
    # Display the selected hotel
    st.write("Bạn đã chọn:", selected_hotel)

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_hotel_id = selected_hotel[1]

    if st.session_state.selected_hotel_id:
        st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
        # Hiển thị thông tin khách sạn được chọn
        hotel_info = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]

        if not hotel_info.empty:
            st.write('#### Bạn vừa chọn:')
            st.write('### ', hotel_info['Hotel_Name'].values[0])

            hotel_description = hotel_info['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:100])
            st.write('##### Thông tin:')
            st.write(truncated_description, '...')

            st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
            recommendations = get_recommendations_content(df_info, hotel_id=st.session_state.selected_hotel_id, top_n=5, content_gem=content_new, dictionary=dict_new, index=index_new, tfidf=tdidf_new) 
        
            st.session_state.recommendations = recommendations
            # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
            hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.recommendations.iterrows()]
            # Tạo một dropdown với options là các tuple này
            selected_hotel = st.radio(
                "Chọn khách sạn",
                options=hotel_options,
                format_func=lambda x: x[0]  # Hiển thị tên khách sạn
        )
            
            st.write("Bạn đã chọn:", selected_hotel)

            # Cập nhật session_state dựa trên lựa chọn hiện tại
            st.session_state.selected_hotel_id = selected_hotel[1]
            if st.session_state.selected_hotel_id:
                # st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
                
                # Hiển thị thông tin khách sạn được chọn
                hotel_info = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]
                
                hotel_description = hotel_info['Hotel_Description'].values[0]
                truncated_description = ' '.join(hotel_description.split()[:200])     
            
                # Displaying the hotel details in a table
                hotel_details = {
                    "Tên": hotel_info['Hotel_Name'].values[0],
                    "ID": st.session_state.selected_hotel_id,
                    "Thông tin": truncated_description + '...',
                    "Địa chỉ": hotel_info['Hotel_Address'].values[0]
                }
                df = pd.DataFrame(hotel_details, index=[0])
                st.dataframe(df,hide_index=True)
                
                hotel_scores = {
                    "Total Score": hotel_info['Total_Score'].values[0],
                    "Location": hotel_info['Location'].values[0],
                    "Cleanliness": hotel_info['Cleanliness'].values[0],
                    "Service": hotel_info['Service'].values[0],
                    "Facilities": hotel_info['Facilities'].values[0],
                    "Value for money": hotel_info['Value_for_money'].values[0],
                    "Comfort and room quality": hotel_info['Comfort_and_room_quality'].values[0],
                }
                for category, score in hotel_scores.items():
                    st.write(f"**{category}:** {score} ⭐")

                    
                st.write('Khách đến từ các nước:')
                nationality_counts = df_comments_full[(df_comments_full['Hotel ID']==st.session_state.selected_hotel_id)]['Nationality'].value_counts().to_dict()
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(nationality_counts)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
                
                plt.figure(figsize=(10, 5))
                sns.histplot(df_comments_full[(df_comments_full['Hotel ID']==st.session_state.selected_hotel_id)]['Score'], bins=10, kde=True)
                plt.title('Điểm số khách sạn')
                plt.xlabel('Score')
                plt.ylabel('Frequency')
                st.pyplot(plt)
                
                average_scores = df_comments_full[(df_comments_full['Hotel ID']==st.session_state.selected_hotel_id)].groupby('Room Type')['Score'].mean().reset_index()
                pd.options.display.float_format = '{:,.2f}'.format
                average_scores['Score'] = average_scores['Score'].apply(lambda x: '{:,.2f}'.format(x))
                st.write("Điểm số trung bình loại phòng")
                st.table(average_scores)
        else:
            st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")
    
elif choice == 'Collaborative Filtering':

    ###### Giao diện Streamlit ######
    st.image('hotel.jpg', use_column_width=True)

    # Dropdown for selecting the user's name
    selected_name = st.selectbox("Chọn tên:",
                                options=df_comments['Reviewer Name'].unique())

    # Dropdown for selecting the user's nationality
    selected_nationality = st.selectbox("Chọn quốc tịch:",
                                        options=df_comments[df_comments['Reviewer Name'] == selected_name]['Nationality'].unique())

    # Find the corresponding user_id
    user_id = df_comments[(df_comments['Reviewer Name'] == selected_name) & (df_comments['Nationality'] == selected_nationality)]['reviewer_id_encoded'].values[0]

    # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
    if 'selected_user_name' not in st.session_state:
        # Nếu chưa có, thiết lập giá trị mặc định là None hoặc ID khách sạn đầu tiên
        st.session_state.selected_user_name = None
        
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None

    # Display the selected hotel
    st.write(f"Bạn đã chọn: {selected_name} - {selected_nationality} - {user_id}")

    # Cập nhật session_state dựa trên lựa chọn hiện tại
    st.session_state.selected_user_name = selected_name
    st.session_state.selected_user_nationality = selected_nationality

    if st.session_state.selected_user_name:
        st.write("Tên: ", selected_name)
        st.write("Quốc tịch: ", selected_nationality)
        st.write("User ID: ", user_id)
        
        # Display the top 5 hotel recommendations
        st.write('##### Các khách sạn khác bạn có thể quan tâm:')
        recommendations = get_recommendations(df_comments,df_info,reviewer_name=st.session_state.selected_user_name,trainset=trainset_new,algorithm=algo_new,hotels_id=hotels_id_new, num=5)
        # display_recommended_hotels(recommendations,cols=3)
        
        st.session_state.recommendations = recommendations
        # Tạo một tuple cho mỗi khách sạn, trong đó phần tử đầu là tên và phần tử thứ hai là ID
        hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.recommendations.iterrows()]
        # Tạo một dropdown với options là các tuple này
        selected_hotel = st.radio(
            "Chọn khách sạn",
            options=hotel_options,
            format_func=lambda x: x[0]  # Hiển thị tên khách sạn
        )
        
        # Display the selected hotel
        st.write("Bạn đã chọn:", selected_hotel)

        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_hotel_id = selected_hotel[1]
        if st.session_state.selected_hotel_id:
            # st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
            
            # Hiển thị thông tin khách sạn được chọn
            hotel_info = df_hotels[df_hotels['Hotel_ID'] == st.session_state.selected_hotel_id]
            
            hotel_description = hotel_info['Hotel_Description'].values[0]
            truncated_description = ' '.join(hotel_description.split()[:200])
            # st.write('Thông tin:')
            # st.write(truncated_description, '...')
            # st.write("Địa chỉ:", hotel_info['Hotel_Address'].values[0])
            # st.write("Điểm số:", hotel_info['Total_Score'].values[0])
            # st.write("Ranking:", hotel_info['Hotel_Rank'].values[0])        
        
            # Displaying the hotel details in a table
            hotel_details = {
                "Tên": hotel_info['Hotel_Name'].values[0],
                "ID": st.session_state.selected_hotel_id,
                "Thông tin": truncated_description + '...',
                "Địa chỉ": hotel_info['Hotel_Address'].values[0]
            }
            df = pd.DataFrame(hotel_details, index=[0])
            st.dataframe(df,hide_index=True)
            
            hotel_scores = {
                "Total Score": hotel_info['Total_Score'].values[0],
                "Location": hotel_info['Location'].values[0],
                "Cleanliness": hotel_info['Cleanliness'].values[0],
                "Service": hotel_info['Service'].values[0],
                "Facilities": hotel_info['Facilities'].values[0],
                "Value for money": hotel_info['Value_for_money'].values[0],
                "Comfort and room quality": hotel_info['Comfort_and_room_quality'].values[0],
            }
            for category, score in hotel_scores.items():
                st.write(f"**{category}:** {score} ⭐")

                
            st.write('Khách đến từ các nước:')
            # Count the occurrences of each nationality
            nationality_counts = df_comments_full[(df_comments_full['Hotel ID']==st.session_state.selected_hotel_id)]['Nationality'].value_counts().to_dict()
            # Generate the word cloud without splitting by spaces
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(nationality_counts)
            # Display the word cloud in Streamlit
            # st.set_option('deprecation.showPyplotGlobalUse', False)  # Avoid a deprecation warning
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
            
            plt.figure(figsize=(10, 5))
            sns.histplot(df_comments_full[(df_comments_full['Hotel ID']==st.session_state.selected_hotel_id)]['Score'], bins=10, kde=True)
            plt.title('Điểm số khách sạn')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            st.pyplot(plt)
            
            average_scores = df_comments_full[(df_comments_full['Hotel ID']==st.session_state.selected_hotel_id)].groupby('Room Type')['Score'].mean().reset_index()
            pd.options.display.float_format = '{:,.2f}'.format
            average_scores['Score'] = average_scores['Score'].apply(lambda x: '{:,.2f}'.format(x))
            # Display the average scores in Streamlit
            st.write("Điểm số trung bình loại phòng")
            st.table(average_scores)
        
    else:
        st.write(f"Không tìm thấy khách lưu trú với tên {st.session_state.selected_user_name} và quốc tịch {st.session_state.selected_user_nationality}")
