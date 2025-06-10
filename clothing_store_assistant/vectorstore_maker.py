import os
import pandas as pd
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import chromadb

# === Load Data ===
df = pd.read_csv("C:\\Users\\badri\\Downloads\\fashion_with_json.csv", engine="python", on_bad_lines="skip")

# === Filter clothing types ===
clothing_article_types = [
    'Blazers', 'Boxers', 'Bra', 'Briefs', 'Camisoles', 'Capris', 'Churidar', 'Clothing Set',
    'Dresses', 'Dupatta', 'Innerwear Vests', 'Jackets', 'Jeans', 'Jeggings', 'Jumpsuit',
    'Kurta Sets', 'Kurtas', 'Kurtis', 'Leggings', 'Lehenga Choli', 'Lounge Pants',
    'Lounge Shorts', 'Lounge Tshirts', 'Nehru Jackets', 'Night suits', 'Nightdress',
    'Patiala', 'Rain Jacket', 'Rain Trousers', 'Robe', 'Rompers', 'Salwar',
    'Salwar and Dupatta', 'Sarees', 'Shapewear', 'Shirts', 'Shorts', 'Shrug', 'Skirts',
    'Suits', 'Sweaters', 'Sweatshirts', 'Swimwear', 'Tights', 'Tops', 'Track Pants',
    'Tracksuits', 'Trousers', 'Tshirts', 'Tunics', 'Waistcoat'
]
df = df[df['articleType'].isin(clothing_article_types)].reset_index(drop=True)
df['id'] = df['id'].astype(str)
df = df.sample(frac=0.6).reset_index(drop=True)

# === Explanation Map for Ambiguity Resolution ===
category_explanation_map = {
  "Tshirts": "a casual, collarless knit top usually with short sleeves.",
  "Clothing Set": "a matching or coordinated combination of garments sold as a set, often including tops and bottoms.",
  "Kurta Sets": "a traditional Indian outfit consisting of a kurta and matching bottoms.",
  "Swimwear": "garments designed specifically for swimming, including swimsuits, bikinis, and trunks.",
  "Capris": "close-fitting pants that extend below the knee but above the ankle, often worn casually.",
  "Churidar": "tight-fitting trousers worn under kurtas, typically longer and gathered at the ankles.",
  "Jeans": "sturdy, casual trousers made from denim fabric, popular for everyday wear.",
  "Jeggings": "leggings styled to resemble jeans, combining comfort with denim aesthetics.",
  "Leggings": "tight-fitting stretch pants, typically ankle-length, worn as activewear or under tunics.",
  "Patiala": "a type of pleated, baggy trousers traditional to Punjab, usually paired with kurtas.",
  "Rain Trousers": "water-resistant or waterproof pants worn to protect from rain.",
  "Salwar and Dupatta": "a combination of loose trousers (salwar) and a long scarf (dupatta), often worn with a kameez or kurta.",
  "Salwar": "a type of loose-fitting trousers gathered at the waist and ankles, often worn with Indian ethnic tops.",
  "Shorts": "casual or athletic lower garments that end above the knees.",
  "Skirts": "lower garments that hang from the waist and flow freely around the legs.",
  "Tights": "tight-fitting leg coverings that extend from the waist to the toes, often worn under skirts or dresses.",
  "Track Pants": "comfortable, athletic-style pants typically made from knit or polyester fabric, used for workouts or lounging.",
  "Tracksuits": "a coordinated set of track pants and jacket, used for sports or athleisure.",
  "Trousers": "formal or semi-formal pants that cover the legs from waist to ankle.",
  "Dresses": "one-piece garments for women or girls that combine a bodice with a skirt.",
  "Jumpsuit": "a one-piece garment combining top and pants, worn casually or formally depending on fabric and design.",
  "Boxers": "loose-fitting innerwear or loungewear, typically worn by men.",
  "Bra": "an undergarment designed to support or cover the breasts.",
  "Briefs": "close-fitting underwear for men or women that provide coverage from waist to upper thighs.",
  "Camisoles": "lightweight, sleeveless upper garments typically used as innerwear or summer tops.",
  "Innerwear Vests": "sleeveless inner garments worn under shirts or kurtas for comfort and sweat absorption.",
  "Shapewear": "form-fitting undergarments designed to smooth or contour the body’s silhouette.",
  "Lounge Pants": "comfortable, loose-fitting pants designed for home or sleepwear.",
  "Lounge Shorts": "short, comfortable bottoms intended for indoor use or lounging.",
  "Lounge Tshirts": "casual t-shirts primarily intended for indoor or sleepwear use.",
  "Night suits": "coordinated sleepwear sets, typically including a top and bottom, designed for bedtime comfort.",
  "Nightdress": "a one-piece sleepwear garment for women, usually made from soft fabrics like cotton or satin.",
  "Robe": "a loose-fitting outer garment, often used for lounging or after bathing.",
  "Sarees": "a traditional Indian garment consisting of a long fabric draped over a blouse and petticoat.",
  "Blazers": "semi-formal or formal upper-body garments with lapels, often part of suits.",
  "Dupatta": "a traditional scarf or shawl worn with Indian ethnic wear, typically draped over shoulders.",
  "Jackets": "outerwear designed for warmth, wind or rain protection, can be casual or formal.",
  "Kurtas": "long, collarless shirts worn in South Asia, suitable for both men and women.",
  "Kurtis": "shorter versions of kurtas, typically worn by women with leggings or jeans.",
  "Lehenga Choli": "a traditional Indian outfit consisting of a flared skirt (lehenga) and a cropped blouse (choli), often worn with a dupatta.",
  "Nehru Jackets": "hip-length tailored coats with a mandarin collar, traditionally worn over kurtas.",
  "Rain Jacket": "a waterproof or water-resistant outer garment designed to keep the upper body dry during rain.",
  "Rompers": "a one-piece garment combining a top and shorts, worn casually or for playwear.",
  "Shirts": "a collared, button-down garment. Can be formal or casual depending on fabric and style.",
  "Shrug": "a cropped cardigan-style outer layer worn over tops or dresses.",
  "Suits": "a formal set of garments consisting of a jacket and trousers or skirt.",
  "Sweaters": "knitted upper-body garments used for warmth.",
  "Sweatshirts": "warm, thick upper garments made from fleece or cotton blends, often used for casual wear or sports.",
  "Tops": "a broad term for women's upper garments not categorized as shirts or t-shirts.",
  "Tunics": "long tops worn by women, typically extending below the hips.",
  "Waistcoat": "a sleeveless, buttoned formal garment worn over a shirt, usually part of a suit."
}



# === Clean description ===
def clean_description(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text(separator=" ").strip()


# === Construct Enriched Text ===
def construct_text(row):
    desc = clean_description(row['description']) if pd.notna(row['description']) else ""
    desc = desc[:100]
    atype = row['articleType']
    extra = category_explanation_map.get(atype, "")

    return (
        f"{row['productDisplayName']}. {desc} "
        f"This product is a {atype}: {extra} "
        f"It is {row['baseColour']} in color, designed for {row['gender']}. "
        f"Best used in {row['usage']} during {row['season']}."
    )


# === Load model ===
text_model = SentenceTransformer('models/all-mpnet-base-v2')

# === Setup ChromaDB ===
client_text = chromadb.PersistentClient(path="./chroma_store_text")
if "ecommerce_text" in [c.name for c in client_text.list_collections()]:
    client_text.delete_collection("ecommerce_text")
collection_text = client_text.get_or_create_collection("ecommerce_text")

# === Build vectorstore ===
for _, row in tqdm(df.iterrows(), total=len(df), desc="Building text-only vectorstore"):
    pid = row['id']
    meta_text = construct_text(row)
    embedding = text_model.encode(meta_text)

    row_meta = row.dropna().to_dict()
    row_meta["category_explanation"] = category_explanation_map.get(row['articleType'], "")

    collection_text.add(
        ids=[pid],
        embeddings=[embedding.tolist()],
        documents=[meta_text],
        metadatas=[row_meta]
    )

print("✅ Text-only vectorstore with disambiguated articleTypes created.")
