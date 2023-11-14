    • Abstract TEAM
    • Introduction
        ◦ Motivation (including Related work) JAY
    • Fundamentals
    • Datasets
        ◦ Mars 2020 Mission Information ASHWIN
            ▪ Cameras (Mast, Navigation) {JAY}
            ▪ Image Data (format, location, naming convention) 
                • Data Type Codes 
                • Available Metadata (Tushar) 
            ▪ Data Processing {Ashwin}
                • Image selection from PDS
                • EBY Images {Ashwin}
        ◦ MRO Mission Information DEM group
            ▪ Cameras (CTX, HiRise)
            ▪ Image Data (format, location, naming convention)
    • Methods
        ◦ Camera Models NADINE
            ▪ Pin Hole Camera model Nadine
            ▪ CAHVORE Camera Model Nadine
            ▪ Intrinsics Nadine
            ▪ Extrinsics (Tushar)
        ◦ Stereo Reconstruction 
            ▪ Stereo Vision Nadine
            ▪ Stereo Geometry Nadine
            ▪ Triangulation Nadine
            ▪ Stereo Matching Nadine
            ▪ Calibrated vs Uncalibrated Reconstruction Jay
            ▪ (SFM) {Ashwin}
            ▪ Ames Pipeline Fundamentals Mohammed
        ◦ Shape from Shading Daniela
        ◦ 3D point clouds to 3D models algorithms MENA
            ▪ Implicit functions, isosurfaces
            ▪ Poisson reconstruction, marching cubes, computing normals
        ◦ VR fundamentals (what we use but did not create) MORITZ
            ▪ Game Engine: Possible options and why we chose Unity, what is provided by it
            ▪ Unity: Engine, Terrain Toolbox
            ▪ SteamVR Plugin: Handling of VR, inputs, teleport movement, Snap turn
            ▪ Shaders/ shaders in Unity ANUSHKA
                • Render pipeline, materials, Default Terrain Shader, …
    • Implementation
        ◦ Reconstruction from Rover Images
            ▪ Matlab {Nadine}
                • intrinsics + extrinsics computation
                • uncalibrated
                • calibrated
            ▪ OpenCV {Jay/Tushar}
                • intrinsics + extrinsics computation 
                • … is there anything else successfully done here?
            ▪ Metashape: {Nadine , Ashwin , Jay, Tushar}
                • Uncalibrated Pipeline {Nadine}
                    ◦ Un-Calibrated Reconstruction Overview in Agisoft For Point Cloud and Mesh Generation 
                    ◦ Alignment Process {Jay} Markers and masks
                    ◦ Dense Point Cloud Formation 
                    ◦ Mesh and Texture Generation
                    ◦ Mesh Optimisation in Agisoft {Jay}
                • Calibrated Pipeline {3D rover Team}
                    ◦ Calibrated Reconstruction Overview in Agisoft For Point Cloud and Mesh Generation {Nadine}
                    ◦ Extrinsic Parameters {TB}
                    ◦ Intrinsics {Jay}
                    ◦ Camera Model Fitting in Agisoft {}
                • Pointcloud stitching & ICP {Ashwin}
                    ◦ Pointcloud alignment & stiching for uncalibrated pointclouds w/ CC & Agisoft {Naden}
                    ◦ Pointcloud alignment & stitching for calibrated pointclouds {Ashwin}
        ◦ DEMs
            ▪ Image Calibration (ISIS)
            ▪ Stereo: Ames Pipeline, Parameters & Settings
            ▪ SfS: Scripts, Atm, Parameter Selection
        ◦ VR
        ◦ Pipeline 1: Importing DEMs into Unity
            ▪ Diagram showing the process: DEM → heightmap/ JSON → Terrain Toolbox Import MORITZ
            ▪ Preprocessing/ Conversion satellite DEMs in Matlab ANUSHKA, MORITZ
                • Requirements: Unity/ Terrain Toolbox needs 16-bit squared heightmaps ANUSHKA
                    ◦ → cropping/ tiling
                • Functions for converting DEMs and retaining auxiliary data MORITZ
            ▪ Import in Unity via the Terrain Toolbox (Settings, e.g. important to use batch mode) ANUSHKA
            ▪ Additional Settings and Scripts: TerrainAttributes MORITZ
                • DEM meta data imorter in Unity (geogrphic coordinates)
                • BaseTerrainHandling (Allows Selection and Teleport to Geographic Coordinates)
            ▪ Texturing/ Shading of the Terrain (instead of images as textures)
                • TerrainLayers for Texturing: Uniform Albedo/ Albedo map MORITZ
                • Lambert/ Other Shaders instead of standard surface shader ANUSHKA
                • (Splatmapping based on slope/ albedo) MORITZ
        ◦ Pipeline 2: Creating 3D Point clouds in Agisoft and 3D models in Meshlab and Cloudcompare MENA
            ▪ Diagram showing the entire process
            ▪ (Creating and aligning Pointclouds)
            ▪ Surface reconstruction with Poisson reconstruction
            ▪ Remeshing, Decimation, Texture & Normal maps & Export
        ◦ Extension: Importing 3D models in Unity and setting them in the scene 
            ▪ Import .fbx and textures 
            ▪ Find approximate location and embed in Terrain (SfS HiRise)
        ◦ Unity: Additional features (created from scratch or adapted from tutorials) MORITZ
            ▪ Additional movement options: Smooth Movement and Flying
            ▪ UI with Settings, Terrain Selection, Teleport functionality, Laserpointer interaction
            ▪ Interactive “Lab” with DEM showcases under different illumination conditions
    • Results/ Discussion
        ◦ Rover group
            ▪ Disparity Map output Matlab {Nadine}
            ▪ Point cloud {Nadine}/ Stitched point cloud -tbc {ASHWIN}
            ▪ Disparity Map output in open CV {Jay}
            ▪ PC output Open CV -{Jay,Tushar}
        ◦ DEMs
            ▪ CTX 
                • Stereo
                • SfS
            ▪ HiRise
                • Stereo 
                • SfS
        ◦ VR
            ▪ 3D meshes for the Pointclouds MENA
                • Description/ Properties: Quality, …
            ▪ Terrains 
                • Scale, Quality, Evaluating the impression in VR MORITZ
                • (Textures)
            ▪ Scene: Jezero crater with DEMs and 3D models from Preserverance images MORITZ
                • Providing images/ videos of the experience
    • Conclusion and Outlook
        ◦ Limitations 
            ▪ Calibration plate , Matlab calibration tool {NE, JAY}// {3D Reconstruction Team}
        ◦ (Usefulness of VR visualization of planetary data)
        ◦ Open Questions
        ◦ Future Work {Tushar}
p
