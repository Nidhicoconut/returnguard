# Amazon demo orders

Each demo customer's orders are real Amazon Fashion products. `catalog/<SKU>/` holds the product's official listing photos (the SKU is the product's parent ASIN); `customer_photos/` holds a real customer photo of that product arriving damaged. The orders themselves are defined in `app/db.py`.

**Research use only:** Amazon Reviews 2023 (McAuley Lab, Hou et al., 2024). Keep this folder local (it is gitignored); don't publish or redistribute the photos.

## Try it

Sign in as the customer, start a return on the order, describe the damage and upload the photo. The product check compares the photo with the order (type and colour, via Gemini) and with the catalog photos; the damage check and the second vision check then run as usual.

| Order | Sign in as | Product (colour bought) | Photo to upload | What the customer wrote |
|---|---|---|---|---|
| ORD-3001 | asha / demo1234 | Women's V-Neck Crop Sweater (white) | `customer_photos/ORD-3001_asha_upload.jpg` | "Came with a hole at the seam low quality" |
| ORD-3002 | asha / demo1234 | SheIn Mock Neck Ribbed Knit Tee (army green) | `customer_photos/ORD-3002_asha_upload.jpg` | "Arrived damaged due to poor quality." |
| ORD-3003 | ravi / demo1234 | Cable Knit Sweater Set (hoodie crop top and sweatpants) (black) | `customer_photos/ORD-3003_ravi_upload.jpg` | "UPDATE!!! This is cute but JUNK…SUPER CUTE." |
| ORD-3004 | ravi / demo1234 | Zexxxy Striped Short-Sleeve Pyjama Set (grey) | `customer_photos/ORD-3004_ravi_upload.jpg` | "Cheap Fabric" |
| ORD-3005 | neha / demo1234 | Hue Chevron Quilted Control-Top Tights (navy) | `customer_photos/ORD-3005_neha_upload.jpg` | "Beautiful tights but item arrived damaged :(" |
| ORD-3006 | neha / demo1234 | Delia's Girls Thermal Knit Shirt and Pant Set, 2-pack (lavender and pink) | `customer_photos/ORD-3006_neha_upload.jpg` | "Hole after 1st Use" |

To show a **wrong item**: upload one order's photo against a different kind of order, for example the tights photo (`ORD-3005_neha_upload.jpg`) on Asha's sweater order ORD-3001. The app should ask for a photo of the right item, and after the photo requests are used up send the case to a reviewer as "Product mismatch".

Notes:
- The customer photos are close-ups of the damage. A catalog listing photo of a whole garment on a model looks very different, so catalog similarity mostly stays below the 0.85 "match" line; the Gemini type and colour check does the identification (see `CV/datasets/amazon_catalog_test/RESULTS.md`).
- Prices and delivery dates are made up for the demo; the products, listing photos and customer photos are real.
