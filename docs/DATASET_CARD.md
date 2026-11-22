# Dataset Card: SKU-110K & Grozi-120 Retail Shelf Subset

- **Source**: Curated 1,200-image subset from SKU-110K (Goldman et al., CVPR 2019) and Grozi-120 (Merler et al., CVPRW 2007).
- **Annotations**: 4,850 bounding boxes across `product_sku` (74.2%), `stockout_gap` (15.8%), and `shelf_rail` (10.0%).
- **Partition Strategy**: 70% Train (840) / 15% Validation (180) / 15% Test (180).
- **Leakage Prevention**: Aisle-level group splitting, perceptual hash (pHash) near-duplicate removal, and train-only feature standardization.