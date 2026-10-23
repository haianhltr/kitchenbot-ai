# Third-party assets

Generated from `scripts/assets_manifest.yaml` by `python scripts/fetch_assets.py --write-manifest-md`; do not edit by hand. The manifest and this file are committed, the files themselves are git-ignored and fetched with `python scripts/fetch_assets.py --set dishwasher|cabinets|all`.

## Sources

| Source | URL | Pinned revision | Licence | Reference |
| --- | --- | --- | --- | --- |
| ArtVIP (`artvip`) | <https://huggingface.co/datasets/x-humanoid-robomind/ArtVIP> | `d22f2095bc9db29fdd4b60f4a3c4f8177e1f5e9e` | Apache-2.0 | arXiv 2506.04941 |
| MolmoSpaces (Ai2) (`molmospaces`) | <https://huggingface.co/datasets/allenai/molmospaces> | `5f802a46816ffc3520e59b5ef49fa8dafee8eeca` | CC-BY-4.0 (data, except the Objaverse subset: ODC-BY-1.0); code Apache-2.0 | arXiv 2602.11337 |

Licence notes. ArtVIP: the Hugging Face dataset card metadata says `apache-2.0`; the repo ships no separate LICENSE file. MolmoSpaces: the card says all subsets except Objaverse are CC BY 4.0 (attribution required: Ai2, MolmoSpaces, arXiv 2602.11337) and adds that the data is intended for research and educational use under Ai2's Responsible Use Guidelines. The `commercial_safe` filter exists only in the code (<https://github.com/allenai/molmospaces>, an environment variable that drops non-commercial Objaverse assets); the dataset card has no such flag and the THOR assets used here are not Objaverse.

## Assets

| Set | Asset | Source | Files | Size | Licence | What we use it for |
| --- | --- | --- | ---: | ---: | --- | --- |
| dishwasher | `dishwasher_1` | artvip | 8 | 20.0 MB | Apache-2.0 | dishwasher candidate for #11 (inspected: side-hinged door, 2 racks, 0.2 m travel; not chosen) |
| dishwasher | `dishwasher_2` | artvip | 6 | 7.0 MB | Apache-2.0 | recommended dishwasher for #11: X-axis door 0-90 deg, two prismatic racks, lightest mesh |
| dishwasher | `dishwasher_3` | artvip | 8 | 5.9 MB | Apache-2.0 | dishwasher candidate for #11 (inspected: shallow racks 0.21 m deep; not chosen) |
| dishwasher | `dishwasher_4` | artvip | 8 | 39.2 MB | Apache-2.0 | alternate dishwasher for #11: ships door open and racks out, but 264k points |
| dishwasher | `l_dishwasher_1` | artvip | 6 | 1.3 MB | Apache-2.0 | inspected: drawer-type dishwasher, two drawers and no door; not chosen |
| dishwasher | `l_dishwasher_2` | artvip | 7 | 4.3 MB | Apache-2.0 | inspected: drawer-type dishwasher, two drawers and no door; not chosen |
| dishwasher | `l_dishwasher_3` | artvip | 8 | 3.7 MB | Apache-2.0 | inspected: drawer-type dishwasher, two drawers and no door; not chosen |
| cabinets | `cupboard_1` | artvip | 16 | 46.5 MB | Apache-2.0 | inspected: whole kitchen run, 21 joints, 729k points, sdf collision; reference only |
| cabinets | `cupboard_2` | artvip | 14 | 20.1 MB | Apache-2.0 | inspected: whole kitchen run, 22 joints, 698k points, sdf collision; reference only |
| cabinets | `counter` | artvip | 5 | 0.6 MB | Apache-2.0 | recommended counter block with 13 hinged doors for #12 (33k points) |
| cabinets | `kitchen_cupboard` | artvip | 5 | 6.2 MB | Apache-2.0 | recommended base cabinet with 9 drawers and 1 door for #12 (4.7k points) |
| cabinets | `thor_Cabinet` | molmospaces | 1 | 0.2 MB | CC-BY-4.0 | static kitchen cabinet variants with primitive and convex colliders, filler dressing for #12 |
| cabinets | `thor_Countertop` | molmospaces | 1 | 1.1 MB | CC-BY-4.0 | static countertop variants, filler dressing for #12 |

Total: 93 downloaded files, 156.2 MB (cap 2 GB). MolmoSpaces packs objects into tar shards: each `.tar.zst` is fetched by HTTP byte range from the shard named in the YAML and unpacked next to the archive.

## Files

Paths are relative to `source/kitchenbot/scene/assets/third_party/`. `sha256` is the first 16 hex digits; the full value is in the YAML.

| Set | File | Size (bytes) | sha256 | Licence |
| --- | --- | ---: | --- | --- |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_1/Isaac Sim 4.5/isaac-sim-assets/Materials/2023_2_1/Base/Glass/Clear_Glass.mdl` | 1177 | `5454a44463d100c6` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_1/model_dishwasher_1.usd` | 8192957 | `89b6fc28c9d77559` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_1/resource/dishwasher_control.py` | 7663 | `3c6501700dafc10d` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_1/resource/img/65a87ebc42ba0bf5c8e6dbf6dbc0fd07.png` | 4723 | `058f07158c0f5ae3` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_1/resource/img/85c0eb6cc1bb6b155c2e47c605366360.png` | 4723 | `058f07158c0f5ae3` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_1/resource/img/f15c057de2aff770622ceae069003627.png` | 4723 | `058f07158c0f5ae3` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_1/resource/material.usd` | 3424 | `ce4c38ead6c9080a` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_1/resource/v_0/link_default_0.usd` | 11824282 | `176ba15634b40288` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_2/.collect.mapping.json` | 383 | `22679f364e01a398` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_2/NVIDIA/Materials/2023_2_1/Base/Glass/Clear_Glass.mdl` | 1177 | `5454a44463d100c6` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_2/model_dishwasher_2.usda` | 2712354 | `705cd1fb79c5a13a` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_2/resource/dishwasher_control.py` | 9102 | `399260e9648dc3c1` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_2/resource/material.usda` | 7001 | `b60a50b12bcf6df5` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_2/resource/v_0/link_default_0.usda` | 4293061 | `bc1b72bb0467ba68` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_3/Isaac Sim 4.5/isaac-sim-assets/Materials/2023_2_1/Base/Glass/Clear_Glass.mdl` | 1177 | `5454a44463d100c6` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_3/model_dishwasher_3.usd` | 2644174 | `77774962e9af1319` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_3/resource/dishwasher_control.py` | 9102 | `32e01994e9bb1256` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_3/resource/img/458cb765ee01c928272b3b1084a1fd84.png` | 4723 | `feba5d9029445949` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_3/resource/img/ac8104b144314d7b5fcc4be16e4a1e14.png` | 6934 | `109eba8e66e9c581` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_3/resource/img/da6f6636bf2d896281959e07483a55d8.png` | 4723 | `54df05e00e10796d` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_3/resource/material.usd` | 3809 | `8458f0123e507a13` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_3/resource/v_0/link_default_0.usd` | 3199980 | `ba5f70f6ccd3abd1` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_4/.collect.mapping.json` | 1499 | `88ad08bb21ac9d19` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_4/NVIDIA/Materials/2023_2_1/Base/Glass/Clear_Glass.mdl` | 1177 | `5454a44463d100c6` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_4/model_dishwasher_4.usda` | 14558550 | `7fb86550a141cea7` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_4/resource/img/19d45214e7dfce713d92f8d574a45765.png` | 184008 | `b22c4e507974301e` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_4/resource/img/3e68fb9702e6e02782e7cd3a96e5d6f2.png` | 4722 | `299135556a8f9744` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_4/resource/img/5c497d9e00da966e8c0562622e803278.png` | 4723 | `81ab12430b0b7596` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_4/resource/material.usda` | 12953 | `a87d389562463ea0` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/dishwasher_4/resource/v_0/link_default_0.usda` | 24457963 | `4daf806d3a33d3d4` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_1/model_l_dishwasher_1.usd` | 14315 | `40ecc2d3c37df1e8` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_1/resource/dishwasher_control.py` | 7143 | `a58fe7ae4b7f19a5` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_1/resource/img/1735d74bc2a9c15c57ab75648107f73c.png` | 22347 | `ad69d4a7605ed19b` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_1/resource/material.usd` | 2464 | `d35d021391c3529f` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_1/resource/v_0/element_path.json` | 95 | `9894f9f87da09f04` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_1/resource/v_0/link_default_0.usd` | 1215723 | `769e49dfecb24004` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_2/model_l_dishwasher_2.usd` | 16398 | `757aa10daa565817` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_2/resource/dishwasher_control.py` | 7143 | `3e57c37d021097ec` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_2/resource/img/1735d74bc2a9c15c57ab75648107f73c.png` | 22347 | `ad69d4a7605ed19b` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_2/resource/img/21591064c9e753732b0ed74166479a0c.png` | 17011 | `a4832bf47c2d8c6c` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_2/resource/material.usd` | 2942 | `54c8e9437937d44d` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_2/resource/v_0/element_path.json` | 209 | `6a65436cdb3c1f30` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_2/resource/v_0/link_default_0.usd` | 4206361 | `dc020e5f2d981ce0` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_3/model_l_dishwasher_3.usda` | 418265 | `41e118dae17f0099` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_3/resource/dishwasher_control.py` | 7143 | `2d861a68862e80c0` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_3/resource/img/1735d74bc2a9c15c57ab75648107f73c.png` | 22347 | `ad69d4a7605ed19b` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_3/resource/img/525890862a5b0a0a4bb0679b6f3733e8.png` | 25429 | `787de61778296ef8` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_3/resource/img/55046af30434f5546a0208e96580a38a.png` | 16503 | `b68b2b852cb583aa` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_3/resource/material.usda` | 10569 | `de945f0f4b4dc2cc` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_3/resource/v_0/element_path.json` | 93 | `1dd47ac03c683a9e` | Apache-2.0 |
| dishwasher | `artvip/Articulated_objects/major_appliances/dishwasher/l_dishwasher_3/resource/v_0/link_default_0.usda` | 3171286 | `bb835b331ea11217` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/.collect.mapping.json` | 3878 | `8c9dedf5ab02aae4` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/Isaac Sim 4.5/isaac-sim-assets/Materials/2023_2_1/Base/Glass/Clear_Glass.mdl` | 1177 | `5454a44463d100c6` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/model_cupboard_1.usd` | 23428543 | `5d835c7b29828cff` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/cupboard_control.py` | 9091 | `ece6b0f09e5a01c9` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/00d3552442dcaca20cc8f1a455bb576f.png` | 17011 | `df3f616d1d8486e3` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/2c4a5fc66e02311804c69a2b8c528519.png` | 51977 | `1cd7203264109e82` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/64aadc1eb199bad92ec7119ee5c5fcbd.png` | 144161 | `1145d74acb3eb636` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/8c4d501e06d6abdc7fa9b8b13c1e2348.png` | 160543 | `cbc99bf9b86503b8` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/ab4d39615426734ae4a48f9ed08ce956.png` | 17011 | `950fdd8fb05e23c9` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/b51bc6d7df590bdb3ab9a8c9547ca7cf.png` | 1477 | `1715739f7ebb308c` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/c29c4991c70eb9dad24bf0abf2f9f3f5.png` | 142560 | `9419a6491b4a9696` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/ce2d864770f7da59e8d912b997419174.png` | 5984 | `6c6f14a3f1230f49` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/img/ffb595afe3e8c00020340c3bb652ac81.png` | 17011 | `37b61f0d237938e7` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/mat_90.usd` | 1464 | `167bb672c0d86305` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/material.usd` | 6379 | `c928b4e908369b6b` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_1/resource/v_0/link_default_0.usd` | 22541605 | `56afd0205641d0e3` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/.collect.mapping.json` | 3679 | `83de69af82ecaf80` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/NVIDIA/Materials/2023_2_1/Base/Glass/Clear_Glass.mdl` | 1177 | `5454a44463d100c6` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/model_cupboard_2.usd` | 222097 | `a0f96c59c29cf018` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/cupboard_control.py` | 9006 | `1d6f1ac3573d532e` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/img/00d3552442dcaca20cc8f1a455bb576f.png` | 17011 | `df3f616d1d8486e3` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/img/11ec620449fb61c7cc5e9e5c0d76a84b.png` | 17011 | `59bcb24a0be105d2` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/img/8c4d501e06d6abdc7fa9b8b13c1e2348.png` | 160543 | `cbc99bf9b86503b8` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/img/ab4d39615426734ae4a48f9ed08ce956.png` | 17011 | `950fdd8fb05e23c9` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/img/b51bc6d7df590bdb3ab9a8c9547ca7cf.png` | 1477 | `1715739f7ebb308c` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/img/bfc6a65629a9cd682f25058e89576168.png` | 32380 | `f75ba9b7d57f1d73` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/img/ce2d864770f7da59e8d912b997419174.png` | 5984 | `6c6f14a3f1230f49` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/img/ffb595afe3e8c00020340c3bb652ac81.png` | 17011 | `37b61f0d237938e7` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/material.usd` | 5289 | `6be0c9b565856ac3` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/cupboard/cupboard_2/resource/v_0/link_default_0.usd` | 19557559 | `5452377619bc60c6` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/counter/counter/model_cupboard.usd` | 29924 | `52260a414f71a164` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/counter/counter/resource/img/f91769dce698c58340e30044fea518cf.png` | 74447 | `06c0ee196163e954` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/counter/counter/resource/material.usd` | 2672 | `b81eed1d0016e2c4` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/counter/counter/resource/v_0/element_path.json` | 3582 | `477d478100d648ea` | Apache-2.0 |
| cabinets | `artvip/Articulated_objects/large_furniture/counter/counter/resource/v_0/link_0.usd` | 527814 | `e45aaa68cec67317` | Apache-2.0 |
| cabinets | `artvip/Interactive_scene/kitchen/cupboard/model_cupboard.usd` | 38038 | `be73c03de6e7ec9b` | Apache-2.0 |
| cabinets | `artvip/Interactive_scene/kitchen/cupboard/resource/img/a1de9af06c43afe3f4fbdd5d9fcfa79a.png` | 5057722 | `37fbaa161883aa2d` | Apache-2.0 |
| cabinets | `artvip/Interactive_scene/kitchen/cupboard/resource/img/e1ce27d4a18e263cbbe06f765fd4d644.png` | 998966 | `a66f5dca0dc4b1cc` | Apache-2.0 |
| cabinets | `artvip/Interactive_scene/kitchen/cupboard/resource/material.usd` | 2783 | `9da3c48602e4334e` | Apache-2.0 |
| cabinets | `artvip/Interactive_scene/kitchen/cupboard/resource/v_69323/link_0_0.usd` | 143485 | `0a8177eeddc0b0ee` | Apache-2.0 |
| cabinets | `molmospaces/thor_Cabinet.tar.zst` | 232726 | `f399c07b5b8d3575` | CC-BY-4.0 |
| cabinets | `molmospaces/thor_Countertop.tar.zst` | 1120506 | `7a2315862de82735` | CC-BY-4.0 |
