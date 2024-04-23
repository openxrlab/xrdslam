# Evaluate F-score

This is a modified version of the F-score evaluation of 3D meshes provided by [**Thanks and Temples**](https://github.com/isl-org/TanksAndTemples/tree/master/python_toolbox/evaluation).

For improved evaluation realism, this version does not crop or downsample the meshes.

### Prerequisites
The library has been tested with the following dependencies, but should work with other versions as well.

1. Python 3.8.5
2. Open3D 0.9.0
3. Matplotlib 3.3.3

## Installation

1. Clone the repository to your local directory: <pre><code>git clone https://github.com/tfy14esa/evaluate_3d_reconstruction_lib.git</code></pre>
2. If you don't want to provide the full path to the ground truth mesh when you call the evaluation script, you can specify the path to the folder containing the ground truth meshes in a hard coded way (not recommended). Open the file <pre><code>evaluate_3d_reconstruction/evaluate_3d_reconstruction/config.py</code></pre> and revise the path to where you store the ground truth meshes.
3. Then open the file <pre><code>evaluate_3d_reconstruction/evaluate_3d_reconstruction/evaluate_3d_reconstruction.py</code></pre> and revise the shebang at the top to point to the python executable of your virtual environment (this can be useful if you want to execute the evaluation script directly from the command line (see below)).
4. Activate your virtual environment
5. Enter the root folder of the library: <pre><code>cd evaluate_3d_reconstruction_lib</code></pre>
6. Install the library: <pre><code>pip install .</code></pre>

### Usage

The main function of the library is
<pre><code>def run_evaluation(pred_ply, path_to_pred_ply, scene, distance_thresh=0.10, gt_translate_to_zero=False, pred_translate_to_zero=False):
   """Calculates the F-score from a predicted mesh to a reference mesh. Generates
    a directory and fills this with numerical and mesh results.

        Args:
            pred_ply (string): string object to denote the name of predicted mesh (as a .ply file)
            path_to_pred_ply (string): string object to denote the full path to the pred_ply file
            scene (string): string object to denote the scene name (a corresponding ground truth .ply file with the name "scene + .ply" needs to exist)
            distance_threshold (float):
            gt_translate_to_zero (bool): boolean describing whether to translate gt mesh to origin
            pred_translate_to_zero (bool): boolean describing whether to translate predicted mesh to origin
            icp_align (bool): align the recontructed mesh with the gt using ICP
            full_path_to_gt_ply (string): specify full path to ground truth mesh

        Returns:
            None
    """
</code></pre>

The main function can be called in two principled ways:

1. As an executable directly from the command line as:
<pre><code>evaluate_3d_reconstruction.py pred_ply scene full_path_to_gt_ply</code></pre> To achieve this, run <pre><code>chmod +x evaluate_3d_reconstruction.py</code></pre> and export the path to the script in your bashrc-file i.e. add similar to the following to your bashrc: <pre><code># Export path to my python evaluate 3d reconstruction script
export PATH="/cluster/project/cvl/esandstroem/src/late_fusion_3dconv/deps/evaluate_3d_reconstruction/evaluate_3d_reconstruction:$PATH"</code></pre>

2. As a normal function in other python scripts. To achieve this, simply import the function using <pre><code>from evaluate_3d_reconstruction import run_evaluation</code></pre>
