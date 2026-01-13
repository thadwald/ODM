import os
import json
import numpy as np

from opendm import log
from opendm import types
from opensfm.dataset import DataSet

try:
    import cv2
except ImportError:
    cv2 = None


def rotation_from_angle_axis(angle_axis):
    """Convert angle-axis to rotation matrix."""
    return cv2.Rodrigues(np.asarray(angle_axis))[0]

#from opensfm 
def opk_from_rotation(rotation_matrix):
    """Extract OPK angles (radians) from rotation matrix."""
    Rc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R = rotation_matrix.T.dot(Rc)
    
    omega = np.arctan2(-R[1, 2], R[2, 2])
    phi = np.arcsin(np.clip(R[0, 2], -1.0, 1.0))
    kappa = np.arctan2(-R[0, 1], R[0, 0])
    
    return omega, phi, kappa


def angle_diff(a, b):
    """Signed angular difference, handling wrap-around. Result in [-180, 180]."""
    diff = a - b
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


class ODMRotationAnalysisStage(types.ODM_Stage):
    """
    Stage to analyze rotation errors between initial camera orientations
    (from EXIF/XMP metadata) and bundle-adjusted orientations from OpenSfM.
    
    Compares OPK angles directly for clearer per-axis error analysis.
    """
    
    def process(self, args, outputs):
        tree = outputs['tree']
        reconstruction = outputs['reconstruction']
        
        if not reconstruction.is_georeferenced():
            log.ODM_WARNING("Reconstruction is not georeferenced, skipping rotation analysis")
            return
        
        photos = reconstruction.photos
        opensfm_path = tree.opensfm
        
        # Use DataSet to load the reconstruction
        data = DataSet(opensfm_path)
        reconstructions = data.load_reconstruction()
        
        if not reconstructions or len(reconstructions) == 0:
            log.ODM_WARNING("No reconstruction found, skipping rotation analysis")
            return
        
        recon = reconstructions[0]
        
        log.ODM_INFO("=" * 60)
        log.ODM_INFO("ROTATION ERROR ANALYSIS (OPK comparison)")
        log.ODM_INFO("=" * 60)
        
        # Load exif metadata for orientation priors
        exif_data = self._load_exif_data(opensfm_path)
        
        results = []
        omega_errors = []
        phi_errors = []
        kappa_errors = []
        
        for shot_id, shot in recon.shots.items():
            # Get prior OPK from exif (stored in degrees)
            if shot_id not in exif_data:
                continue
            exif = exif_data[shot_id]
            if 'opk' not in exif:
                continue
            
            opk_prior = exif['opk']
            omega_prior = float(opk_prior['omega'])
            phi_prior = float(opk_prior['phi'])
            kappa_prior = float(opk_prior['kappa'])
            
            # Get reconstructed OPK (extract from pose rotation matrix)
            R_recon = shot.pose.get_rotation_matrix()
            omega_recon, phi_recon, kappa_recon = opk_from_rotation(R_recon)
            
            # Convert reconstructed to degrees
            omega_recon_deg = np.degrees(omega_recon)
            phi_recon_deg = np.degrees(phi_recon)
            kappa_recon_deg = np.degrees(kappa_recon)
            
            # Compute per-axis errors
            d_omega = angle_diff(omega_recon_deg, omega_prior)
            d_phi = angle_diff(phi_recon_deg, phi_prior)
            d_kappa = angle_diff(kappa_recon_deg, kappa_prior)
            
            omega_errors.append(d_omega)
            phi_errors.append(d_phi)
            kappa_errors.append(d_kappa)
            
            total_error = np.sqrt(d_omega**2 + d_phi**2 + d_kappa**2)
            
            results.append({
                'shot_id': shot_id,
                'omega_prior': omega_prior,
                'phi_prior': phi_prior,
                'kappa_prior': kappa_prior,
                'omega_recon': omega_recon_deg,
                'phi_recon': phi_recon_deg,
                'kappa_recon': kappa_recon_deg,
                'd_omega': d_omega,
                'd_phi': d_phi,
                'd_kappa': d_kappa,
                'total_error': total_error,
            })
        
        if len(results) == 0:
            log.ODM_WARNING("No shots with OPK data found for rotation analysis")
            return
        
        omega_errors = np.array(omega_errors)
        phi_errors = np.array(phi_errors)
        kappa_errors = np.array(kappa_errors)
        
        # Log summary statistics per axis
        log.ODM_INFO(f"Analyzed {len(results)} shots with initial orientations")
        log.ODM_INFO("")
        log.ODM_INFO("Per-axis error statistics (degrees, reconstructed - prior):")
        log.ODM_INFO("")
        log.ODM_INFO(f"  OMEGA (roll):   mean={np.mean(omega_errors):+7.2f}  std={np.std(omega_errors):6.2f}  "
                    f"min={np.min(omega_errors):+7.2f}  max={np.max(omega_errors):+7.2f}")
        log.ODM_INFO(f"  PHI (pitch):    mean={np.mean(phi_errors):+7.2f}  std={np.std(phi_errors):6.2f}  "
                    f"min={np.min(phi_errors):+7.2f}  max={np.max(phi_errors):+7.2f}")
        log.ODM_INFO(f"  KAPPA (yaw):    mean={np.mean(kappa_errors):+7.2f}  std={np.std(kappa_errors):6.2f}  "
                    f"min={np.min(kappa_errors):+7.2f}  max={np.max(kappa_errors):+7.2f}")
        
        # Log worst offenders by total error
        results_sorted = sorted(results, key=lambda x: x['total_error'], reverse=True)
        log.ODM_INFO("")
        log.ODM_INFO("Top 10 shots with largest total rotation error:")
        for i, r in enumerate(results_sorted[:10]):
            log.ODM_INFO(f"  {i+1}. {r['shot_id']}: total={r['total_error']:.2f}° "
                        f"(Δω={r['d_omega']:+.1f}, Δφ={r['d_phi']:+.1f}, Δκ={r['d_kappa']:+.1f})")
        
        # Write detailed results to file
        output_file = os.path.join(opensfm_path, 'rotation_errors.csv')
        with open(output_file, 'w') as f:
            f.write("shot_id,omega_prior,phi_prior,kappa_prior,")
            f.write("omega_recon,phi_recon,kappa_recon,")
            f.write("d_omega,d_phi,d_kappa,total_error\n")
            for r in results_sorted:
                f.write(f"{r['shot_id']},{r['omega_prior']:.4f},{r['phi_prior']:.4f},{r['kappa_prior']:.4f},")
                f.write(f"{r['omega_recon']:.4f},{r['phi_recon']:.4f},{r['kappa_recon']:.4f},")
                f.write(f"{r['d_omega']:.4f},{r['d_phi']:.4f},{r['d_kappa']:.4f},{r['total_error']:.4f}\n")
        
        log.ODM_INFO(f"")
        log.ODM_INFO(f"Detailed results written to: {output_file}")
        log.ODM_INFO("=" * 60)
    
    def _load_exif_data(self, opensfm_path):
        """Load all exif metadata files. Returns dict mapping shot_id -> exif dict."""
        exif_dir = os.path.join(opensfm_path, "exif")
        result = {}
        
        if not os.path.isdir(exif_dir):
            return result
        
        for fname in os.listdir(exif_dir):
            if fname.endswith('.exif'):
                shot_id = fname[:-5]
                fpath = os.path.join(exif_dir, fname)
                try:
                    with open(fpath, 'r') as f:
                        result[shot_id] = json.load(f)
                except Exception as e:
                    log.ODM_WARNING(f"Could not load exif for {shot_id}: {e}")
        
        return result
