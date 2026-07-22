"""
export_to_staging.py — one-time copy of real (read-only) catalog data into
a personal MongoDB Atlas staging cluster, so semantic search can be built
and proven end-to-end (including real $vectorSearch indexes) without
needing write access on the production database.

Source: MONGO_URI (production, read-only credentials — same ones already in use)
Target: STAGING_MONGO_URI (a personal Atlas cluster you fully own and control)

Products are copied in full (no personal data there). Sellers are copied
with an explicit field whitelist only — the `users` collection also holds
email, phone, password hashes, bank details and KYC documents, none of
which semantic search needs or should ever leave the production database.

This is a staging/proof-of-concept step (see PROJECT_STATUS.md). It does
not touch production and requires no new permissions beyond the read
access already granted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

PROD_URI = os.getenv("MONGO_URI")
STAGING_URI = os.getenv("STAGING_MONGO_URI")

# Only these seller fields are copied — deliberately excludes email, phone,
# password, refreshToken, sellerProfile.bankDetails, sellerProfile.documents,
# sellerProfile.address, and anything else personal/sensitive.
SELLER_FIELDS = {
    "_id": 1,
    "username": 1,
    "categoryType": 1,
    "followerCount": 1,
    "profilePic": 1,
    "sellerStatus": 1,
    "sellerProfile.businessName": 1,
}

# Bits: excludes `comments` (other users' free-text content) and `userId`
# is kept only because it's an opaque id, not personal data itself.
BIT_FIELDS = {
    "_id": 1,
    "title": 1,
    "description": 1,
    "video": 1,
    "thumbnail": 1,
    "hashtags": 1,
    "products": 1,
    "userId": 1,
    "likeCount": 1,
    "viewCount": 1,
    "shareCount": 1,
    "isActive": 1,
    "isTrending": 1,
    "createdAt": 1,
}


def main():
    if not PROD_URI:
        print("MONGO_URI not set — nothing to read from.")
        sys.exit(1)
    if not STAGING_URI:
        print("STAGING_MONGO_URI not set — nothing to write to. "
              "Add your personal Atlas connection string to .env first.")
        sys.exit(1)

    prod = MongoClient(PROD_URI, serverSelectionTimeoutMS=8000)["zatch"]
    # Database name is hardcoded to "zatch" to match app/db.py's get_db(),
    # regardless of what path (if any) is in the staging connection string.
    staging_db = MongoClient(STAGING_URI, serverSelectionTimeoutMS=8000)["zatch"]

    print("Reading from production (read-only) -> writing to your staging Atlas cluster")

    # Products: copied in full, no personal data in this collection.
    products = list(prod.products.find({}))
    if products:
        staging_db.products.delete_many({})
        staging_db.products.insert_many(products)
    print(f"Products copied: {len(products)}")

    # Sellers: whitelist only, no personal/financial/KYC data.
    sellers = list(prod.users.find({"sellerStatus": {"$exists": True}}, SELLER_FIELDS))
    if sellers:
        staging_db.users.delete_many({})
        staging_db.users.insert_many(sellers)
    print(f"Sellers copied (whitelisted fields only): {len(sellers)}")

    # Bits: whitelist only, excludes other users' comment text.
    bits = list(prod.bits.find({}, BIT_FIELDS))
    if bits:
        staging_db.bits.delete_many({})
        staging_db.bits.insert_many(bits)
    print(f"Bits copied (whitelisted fields only): {len(bits)}")

    print("Done. Staging Atlas cluster is ready for the embedding backfill.")


if __name__ == "__main__":
    main()
