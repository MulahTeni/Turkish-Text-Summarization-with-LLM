import streamlit as st
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def summarize(text, model, tokenizer):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors='pt').to(device)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def main():
    st.title('Büyük Dil Modeli ile Türkçe Metin Özetleme')

    st.write("<h6>Melih Tuna İpek</h6>", unsafe_allow_html=True)
    st.write("<h6>20011053</h6>", unsafe_allow_html=True)
    st.write("<h6>tuna.ipek@std.yildiz.edu.tr</h6>", unsafe_allow_html=True)
    st.write("<h4></h4>", unsafe_allow_html=True)
    
    st.write("Metin özetlemede facebook bart modeli kullanılmıştır. \
                \nFarklı eğitim veri seti boyutları kullanılmış olup test etmek için ilgili butonlar kullanılabilir.")
    
    st.markdown("## Eğitilmiş Olan Modeller:")
    st.markdown("- 10000 veri ile eğitilmiş Model 1")
    st.markdown("- 25000 veri ile eğitilmiş Model 2")
    
    model_col1, model_col2, _, _, _ = st.columns(5)
    
    model_name_or_path = "./sm"
    model = BartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
    
    
    if model_col1.button("Model 1"):
        model_name_or_path = "./sm"
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)
        
        
    if model_col2.button('Model 2'):
        model_name_or_path = "./md"
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)

        
        
    st.write("Model seçimi yapılmazsa özetleme işlemi otomatik olarak 'Model 1' ile yapılmaktadır.")
    input_text = st.text_area("Metni buraya girin:", height=300)
    
    if st.button("Özetle"):
        summarized_text = summarize(input_text, model, tokenizer)
        print(summarized_text)
        st.code(summarized_text, language="markdown")
    
    if st.button("Sıfırla"):
        input_text = ""
        model_name_or_path = "./sm"
        model = BartForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
        tokenizer = BartTokenizer.from_pretrained(model_name_or_path)

if __name__ == "__main__":
    main()
