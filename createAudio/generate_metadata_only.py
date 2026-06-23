import os
import numpy as np
from create_test_wavs import Config, create_test_sample_static
from create_data_base import get_timit_speakers

def run_metadata_regeneration():
    config = Config()
    # קביעת ה-Seed המקורי כדי להבטיח שחזור מדויק של הנתונים
    np.random.seed(config.seed)

    # הגדרת נתיבים זהה לחלוטין לסקריפט המקורי
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_path = os.path.dirname(script_dir)
    timit_path = os.path.join(workspace_path, 'data', 'TIMIT')
    output_path = os.path.join(workspace_path, 'data', 'simulated_audio', config.dataset_title)
    
    os.makedirs(output_path, exist_ok=True)
    
    print("סורק את בסיס הנתונים של TIMIT...")
    male_speakers, female_speakers = get_timit_speakers(timit_path)
    
    print(f"\nמשחזר מטא-דאטה עבור {config.num_samples} מדגמים...")
    print("שימו לב: קבצי ה-WAV הקיימים לא ישתנו.")
    print("-" * 60)
    
    for i in range(config.start_idx, config.start_idx + config.num_samples):
        print(f"\rמעבד מדגם {i}/{config.start_idx + config.num_samples - 1}...", end="", flush=True)
        
        # הרצת הסימולציה - תייצר גיאומטריה זהה לחלוטין לריצה המקורית בגלל ה-Seed
        result = create_test_sample_static(i, config, male_speakers, female_speakers, verbose=False)
        
        # שמירת קובץ ה-metadata בלבד עם השדות החדשים, תוך דריסת קובץ ה-npz הישן והחלקתי
        np.savez(
            os.path.join(output_path, f'metadata_{i}.npz'),
            room_dim=result['room_dim'],
            T60=result['T60'],
            SNR_diffuse=result['SNR_diffuse'],
            mic_positions=result['mic_positions'],
            true_sector_spk1=result['true_geom_sector_first'],
            true_sector_spk2=result['true_geom_sector_second']
        )
    
    print("\n\nשחזור קבצי המטא-דאטה הושלם בהצלחה!")
    print(f"הקבצים המעודכנים נשמרו בנתיב: {output_path}")

if __name__ == "__main__":
    run_metadata_regeneration()