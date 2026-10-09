# Pathological Gait Datasets
Skeleton datasets for Normal, Antalgic, Stiff legged, Lurching, Steppage, and Trendelenburg gaits.

<img width=900 src="https://user-images.githubusercontent.com/4926634/89141517-c754ae80-d57f-11ea-94c0-08650fb902bd.PNG" title="Skeleton">

Data Quantity
-------------------
10 people x 6 gaits x 120 instances

Data Form
-------------------
time, 0, joint0_x, joint0_y, joint0_z, 1, joint1_x, joint1_y, joint1_z, 2, joint2_x, joint2_y, joint2_z, ...

Data Collection
-------------------
We collected datasets by using multiple Kinect system(6 Kinect v2). We calibrated the coordinate systems of all sensors by using ArUco Markers. For more information, please refer to the below reference or contact me.

<img width=900 src="https://user-images.githubusercontent.com/4926634/89141548-d2a7da00-d57f-11ea-8a69-0bcee1d5dc6b.PNG" title="Data Collection">

Reference
-------------------
K. Jun, Y. Lee, S. Lee, D. Lee and M. S. Kim, "Pathological Gait Classification Using Kinect v2 and Gated Recurrent Neural Networks," in IEEE Access, vol. 8, pp. 139881-139891, 2020, doi: 10.1109/ACCESS.2020.3013029.

Contact
-------------------
kooksung930@gm.gist.ac.kr

Parquet preprocessing
---------------------
Download the dataset from
https://github.com/kooksung/pathological_gait_datasets and place the
`Pathological_Gaits` directory at `data/Pathological_Gaits/`.

Build balanced Parquet files by retaining 8 reproducibly selected walk sets
per subject and gait category:

```powershell
e:/ML/Lab/Gait_Abnormality_Project/.venv/Scripts/python.exe src/build_parquet.py
```

The command writes `normal.parquet`, `antalgic.parquet`, `stiff_legged.parquet`,
`lurch.parquet`, `steppage.parquet`, and `trendelenburg.parquet` to
`data/parquet/`, along with `selection_manifest.csv` and `summary.csv`.
The source files are tab-delimited and contain six sensor files per walk set;
all six are retained. The script removes repeated joint-ID columns and stores
the 25 joint coordinates as `float32`.

