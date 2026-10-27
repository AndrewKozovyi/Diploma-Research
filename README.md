# Disclaimer: image sources

The images used to train the model in this repository are **not owned by the author of this repository**.
They are photos of museum specimens published by their holding institutions:

- **Yale Peabody Museum of Natural History**, Vertebrate Paleontology collection, obtained through [LUX: Yale Collections Discovery](https://lux.collections.yale.edu).
- **GBIF (Global Biodiversity Information Facility)**: images attached to fossil specimen occurrence records published by museums and institutions worldwide, obtained through the [GBIF Occurrence API](https://www.gbif.org/developer/occurrence).

All rights to the images belong to their respective institutions and photographers.

## Attribution

The file name of each image identifies its source record:

| File name pattern | Source record |
|---|---|
| `yale_<uuid>_...` | `https://lux.collections.yale.edu/view/object/<uuid>` |
| `parsed_<key>_...` or `<key>_...` | `https://www.gbif.org/occurrence/<key>` |

The record page lists the publishing institution, the catalogue number and the license or rights statement for its media.
GBIF media are published under licenses chosen by each publisher (CC0, CC BY 4.0 or CC BY-NC 4.0).
Yale gives rights information for each object on its LUX record.

When using this data, please cite the original institutions and **GBIF.org**.

## Terms of use

- The images are used **only for non-commercial academic research** as part of a diploma project.
- The license of this repository applies **to the source code only**, not to the images.
- The images are not provided for reuse. Anyone who wants to use them must get them from the original sources and follow each record's license.
- The trained model is used only to classify photos submitted by users of the Dino Predictor application.

## Removal requests

If you are a rights holder and want an image removed, please open an issue in this repository. It will be removed.
