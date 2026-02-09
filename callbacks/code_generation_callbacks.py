"""
Code Generation & Deployment Callbacks
Handles model code generation, compilation, and flashing to embedded devices
"""

from dash import Input, Output, State, no_update, html, ctx
import json
import os
import glob
import subprocess
import serial.tools.list_ports
from pathlib import Path
import traceback

from config.config import MODELS_DIR, PERSISTENT_DIR, get_model_path, get_models_metadata_path
from utils.model_training import EdgeMLModel
from deployment import (generate_deployment_code, generate_and_save_deployment_code,
                        validate_before_deployment)


def compile_with_arduino_cli(code_data, temp_dir, board_fqbn, serial_port, should_upload, verbose):
    """
    Compile using Arduino CLI.
    """
    try:
        # Create sketch directory
        sketch_name = code_data.get(
            'filename', 'sketch.ino').replace('.ino', '')
        sketch_dir = os.path.join(temp_dir, sketch_name)
        os.makedirs(sketch_dir, exist_ok=True)

        # Save .ino file
        sketch_path = os.path.join(sketch_dir, f"{sketch_name}.ino")
        with open(sketch_path, 'w') as f:
            f.write(code_data['code'])

        # Compile
        compile_cmd = ['arduino-cli', 'compile', '--fqbn', board_fqbn]
        if verbose:
            compile_cmd.append('--verbose')
        compile_cmd.append(sketch_dir)

        result = subprocess.run(
            compile_cmd, capture_output=True, text=True, timeout=120)
        output = f"""Arduino CLI Compilation
{'='*50}

Sketch: {sketch_name}.ino
Board: {board_fqbn}

{result.stdout}
{result.stderr}
"""

        if result.returncode != 0:
            return f"{output}\n\n❌ Compilation failed!", False

        # Upload if requested
        if should_upload:
            upload_cmd = ['arduino-cli', 'upload', '-p',
                          serial_port, '--fqbn', board_fqbn, sketch_dir]
            if verbose:
                upload_cmd.insert(2, '--verbose')

            upload_result = subprocess.run(
                upload_cmd, capture_output=True, text=True, timeout=60)
            output += f"\n\n{'='*50}\nUpload to {serial_port}\n{'='*50}\n\n{upload_result.stdout}\n{upload_result.stderr}"

            if upload_result.returncode != 0:
                return f"{output}\n\n❌ Upload failed!", False

            output += "\n\n✅ Compile & Flash successful!"
        else:
            output += "\n\n✅ Compilation successful!"

        return output, True

    except subprocess.TimeoutExpired:
        return "❌ Compilation timeout (>120s)", False
    except FileNotFoundError:
        return "❌ Arduino CLI not found. Please install from https://arduino.github.io/arduino-cli/", False
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", False


def compile_with_platformio(code_data, temp_dir, serial_port, should_upload, verbose):
    """
    Compile using PlatformIO.
    """
    try:
        # Create PlatformIO project structure
        src_dir = os.path.join(temp_dir, 'src')
        os.makedirs(src_dir, exist_ok=True)

        # Save main.cpp (rename .ino to .cpp for PlatformIO)
        sketch_name = code_data.get(
            'filename', 'sketch.ino').replace('.ino', '')
        main_path = os.path.join(src_dir, 'main.cpp')
        with open(main_path, 'w') as f:
            f.write(code_data['code'])

        # Save platformio.ini
        ini_path = os.path.join(temp_dir, 'platformio.ini')
        platformio_config = code_data.get('platformio_ini', '')
        if serial_port:
            # Add upload port to config
            platformio_config += f"\nupload_port = {serial_port}\nmonitor_port = {serial_port}\n"

        with open(ini_path, 'w') as f:
            f.write(platformio_config)

        # Compile
        compile_cmd = ['pio', 'run', '-d', temp_dir]
        if verbose:
            compile_cmd.append('-v')

        result = subprocess.run(
            compile_cmd, capture_output=True, text=True, timeout=180)
        output = f"""PlatformIO Compilation
{'='*50}

Project: {sketch_name}
Directory: {temp_dir}

{result.stdout}
{result.stderr}
"""

        if result.returncode != 0:
            return f"{output}\n\n❌ Compilation failed!", False

        # Upload if requested
        if should_upload:
            upload_cmd = ['pio', 'run', '-d', temp_dir, '--target', 'upload']
            if verbose:
                upload_cmd.append('-v')

            upload_result = subprocess.run(
                upload_cmd, capture_output=True, text=True, timeout=60)
            output += f"\n\n{'='*50}\nUpload to {serial_port}\n{'='*50}\n\n{upload_result.stdout}\n{upload_result.stderr}"

            if upload_result.returncode != 0:
                return f"{output}\n\n❌ Upload failed!", False

            output += "\n\n✅ Compile & Flash successful!"
        else:
            output += "\n\n✅ Compilation successful!"

        return output, True

    except subprocess.TimeoutExpired:
        return "❌ Compilation timeout (>180s)", False
    except FileNotFoundError:
        return "❌ PlatformIO CLI not found. Please install from https://platformio.org/install", False
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n{traceback.format_exc()}", False


def generate_platformio_config(target_board, model_filename, board_name):
    """
    Generate platformio.ini configuration for the selected board.
    """
    # Map Arduino FQBNs to PlatformIO boards
    board_mapping = {
        'm5stack:esp32:m5stick_c': {
            'platform': 'espressif32',
            'board': 'm5stick-c',
            'framework': 'arduino',
            'lib_deps': ['m5stack/M5StickCPlus2']
        },
        'esp32:esp32:esp32': {
            'platform': 'espressif32',
            'board': 'esp32dev',
            'framework': 'arduino',
            'lib_deps': []
        },
        'esp32:esp32:esp32s3': {
            'platform': 'espressif32',
            'board': 'esp32-s3-devkitc-1',
            'framework': 'arduino',
            'lib_deps': []
        },
        'arduino:avr:uno': {
            'platform': 'atmelavr',
            'board': 'uno',
            'framework': 'arduino',
            'lib_deps': []
        },
        'arduino:avr:nano': {
            'platform': 'atmelavr',
            'board': 'nanoatmega328',
            'framework': 'arduino',
            'lib_deps': []
        },
        'STM32:stm32:GenF4': {
            'platform': 'ststm32',
            'board': 'genericSTM32F407VET6',
            'framework': 'arduino',
            'lib_deps': []
        },
        'seeed:nrf52:xiaonRF52840Sense': {
            'platform': 'nordicnrf52',
            'board': 'xiaonRF52840Sense',
            'framework': 'arduino',
            'lib_deps': ['sparkfun/SparkFun LSM6DS3 Breakout']
        }
    }

    config = board_mapping.get(target_board, {
        'platform': 'unknown',
        'board': 'unknown',
        'framework': 'arduino',
        'lib_deps': []
    })

    lib_deps_str = '\n    '.join(
        config['lib_deps']) if config['lib_deps'] else ''
    lib_deps_section = f"lib_deps = \n    {lib_deps_str}" if lib_deps_str else ""

    return f"""; PlatformIO Project Configuration File for {board_name}
; Auto-generated for HAR Model: {model_filename.split('.')[0]}
;
; Build: pio run
; Upload: pio run --target upload
; Monitor: pio device monitor

[env:har_model]
platform = {config['platform']}
board = {config['board']}
framework = {config['framework']}
monitor_speed = 115200
upload_speed = 921600
{lib_deps_section}
"""


def register_callbacks(app):
    """Register all callbacks with the app."""
    @app.callback(
        Output('deployment-model-selector', 'options'),
        Input('tabs', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=False
    )
    def populate_model_selector(tab, base_dir):
        """
        Populate model selector with available trained models from trained_models.json.
        Uses the working directory from the store.
        """
        # Only populate when on the code generation tab to avoid unnecessary loads
        if tab != 'tab-5':  # Code Generation tab
            return no_update

        try:
            # Use stored base directory or default to PERSISTENT_DIR
            if not base_dir:
                base_dir = PERSISTENT_DIR

            models_dir = os.path.join(base_dir, 'models')
            models_metadata_file = os.path.join(
                models_dir, 'trained_models.json')

            if not os.path.exists(models_metadata_file):
                print(
                    f"WARNING: trained_models.json not found at {models_metadata_file}")
                return []

            with open(models_metadata_file, 'r') as f:
                models_metadata = json.load(f)

            options = []
            for model_filename, metadata in models_metadata.items():
                # Create readable label with model type, timestamp, and accuracy
                model_type = metadata.get('model_type', 'unknown')
                train_acc = metadata.get('train_accuracy', 0) * 100
                test_acc = metadata.get('test_accuracy', 0) * 100
                val_acc = metadata.get('val_accuracy', 0) * 100
                classes = metadata.get('classes', 0)
                timestamp = metadata.get('timestamp', '')

                # Extract model name and timestamp from filename (format: modeltype_timestamp.pkl)
                model_name = model_filename.replace(
                    '.pkl', '').replace('.joblib', '')

                # Format: "ModelName (timestamp) | Train: X% | Test: Y% | Val: Z% | N classes"
                label = f"{model_name} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}% | Val: {val_acc:.1f}% | {classes} classes"
                options.append({'label': label, 'value': model_filename})

            # Sort by test accuracy (descending)
            options.sort(key=lambda x: x['label'], reverse=True)

            print(f"Loaded {len(options)} models from {models_metadata_file}")
            return options

        except Exception as e:
            print(f"Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return []

    @app.callback(
        [Output('model-info-display', 'children'),
         Output('model-parameters-display', 'children')],
        Input('deployment-model-selector', 'value'),
        State('working-directory-store', 'data')
    )
    def display_model_info(model_filename, base_dir):
        """
        Display information about the selected model and auto-populate parameters from metadata.
        Uses the working directory from the store.
        """
        if not model_filename:
            no_model_msg = html.Div("No model selected", style={
                                    'color': '#999', 'font-style': 'italic'})
            return no_model_msg, "Select a model to view parameters"

        try:
            # Use stored base directory or default to PERSISTENT_DIR
            if not base_dir:
                base_dir = PERSISTENT_DIR

            models_dir = os.path.join(base_dir, 'models')
            models_metadata_file = os.path.join(
                models_dir, 'trained_models.json')

            if not os.path.exists(models_metadata_file):
                error_msg = html.Div("Metadata file not found",
                                     style={'color': '#dc3545'})
                return error_msg, "Cannot load parameters"

            with open(models_metadata_file, 'r') as f:
                models_metadata = json.load(f)

            metadata = models_metadata.get(model_filename, {})

            # Extract key information
            model_type = metadata.get('model_type', 'unknown').upper()
            train_acc = metadata.get('train_accuracy', 0) * 100
            val_acc = metadata.get('val_accuracy', 0) * 100
            test_acc = metadata.get('test_accuracy', 0) * 100
            features = metadata.get('features', 0)
            classes = metadata.get('classes', 0)

            # Get training parameters from model_params (stored during training)
            model_params = metadata.get('model_params', {})
            sampling_rate = model_params.get(
                'sampling_rate', 100)  # Default 100 Hz
            window_size_ms = model_params.get(
                'window_size_ms', 1500)  # Default 1500 ms

            # Build info display
            info = html.Div([
                html.Div(f"📊 Model: {model_filename.split('.')[0]}", style={
                         'font-weight': 'bold', 'margin-bottom': '5px', 'font-size': '13px'}),
                html.Div(f"Type: {model_type}", style={
                         'margin-bottom': '5px'}),
                html.Div(f"Accuracy: Train {train_acc:.1f}% | Val {val_acc:.1f}% | Test {test_acc:.1f}%", style={
                         'margin-bottom': '5px'}),
                html.Div(f"Features: {features}", style={
                         'margin-bottom': '5px'}),
                html.Div(f"Classes: {classes} activities",
                         style={'color': '#28a745'})
            ])

            # Build parameters display (critical for deployment)
            params = html.Div([
                html.Div([
                    html.Span("📏 Sampling Rate: ", style={
                              'font-weight': 'bold'}),
                    html.Span(f"{sampling_rate} Hz",
                              style={'color': '#2E86AB'}),
                    html.Span(" (from training)", style={
                              'font-size': '11px', 'color': '#999', 'margin-left': '5px'})
                ], style={'margin-bottom': '8px'}),
                html.Div([
                    html.Span("⏱️ Window Size: ", style={
                              'font-weight': 'bold'}),
                    html.Span(f"{window_size_ms} ms",
                              style={'color': '#2E86AB'}),
                    html.Span(f" ({int((window_size_ms / 1000) * sampling_rate)} samples @ {sampling_rate} Hz)",
                              style={'font-size': '11px', 'color': '#999', 'margin-left': '5px'})
                ], style={'margin-bottom': '8px'}),
                html.Div([
                    html.Span("🎯 Feature Count: ", style={
                              'font-weight': 'bold'}),
                    html.Span(f"{features} features",
                              style={'color': '#2E86AB'}),
                    html.Span(" (time + freq domain)",
                              style={'font-size': '11px', 'color': '#999', 'margin-left': '5px'})
                ])
            ])

            return info, params

        except Exception as e:
            print(f"Error displaying model info: {e}")
            import traceback
            traceback.print_exc()
            error_msg = html.Div(f"Error loading model: {str(e)}", style={
                                 'color': '#dc3545'})
            return error_msg, "Cannot load parameters"

    @app.callback(
        Output('toolchain-status', 'children'),
        Input('toolchain-selector', 'value')
    )
    def check_toolchain_status(toolchain):
        """
        Check if selected toolchain is installed and display status.
        """
        try:
            if toolchain == 'arduino':
                # Check Arduino CLI
                result = subprocess.run(['arduino-cli', 'version'],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    return html.Div([
                        html.Span("✅ Arduino CLI detected: ", style={
                                  'font-weight': 'bold', 'color': '#155724'}),
                        html.Span(version)
                    ])
                else:
                    return html.Div([
                        html.Span("⚠️ Arduino CLI not found. ", style={
                                  'font-weight': 'bold', 'color': '#856404'}),
                        html.A("Download here", href="https://arduino.github.io/arduino-cli/",
                               target="_blank", style={'color': '#007bff'})
                    ])
            elif toolchain == 'platformio':
                # Check PlatformIO
                result = subprocess.run(['pio', '--version'],
                                        capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip().split('\n')[0]
                    return html.Div([
                        html.Span("✅ PlatformIO detected: ", style={
                                  'font-weight': 'bold', 'color': '#155724'}),
                        html.Span(version)
                    ])
                else:
                    return html.Div([
                        html.Span("⚠️ PlatformIO not found. ", style={
                                  'font-weight': 'bold', 'color': '#856404'}),
                        html.A("Install via VS Code Extension or CLI", href="https://platformio.org/install",
                               target="_blank", style={'color': '#007bff'})
                    ])
        except FileNotFoundError:
            if toolchain == 'arduino':
                return html.Div([
                    html.Span("❌ Arduino CLI not installed. ", style={
                              'font-weight': 'bold', 'color': '#721c24'}),
                    html.A("Download here", href="https://arduino.github.io/arduino-cli/",
                           target="_blank", style={'color': '#007bff'})
                ])
            else:
                return html.Div([
                    html.Span("❌ PlatformIO not installed. ", style={
                              'font-weight': 'bold', 'color': '#721c24'}),
                    html.A("Install here", href="https://platformio.org/install",
                           target="_blank", style={'color': '#007bff'})
                ])
        except Exception as e:
            return html.Div(f"⚠️ Error checking toolchain: {str(e)}", style={'color': '#856404'})

    @app.callback(
        [Output('serial-port-selector-deploy', 'options'),
         Output('port-info-display', 'children')],
        [Input('refresh-ports-btn', 'n_clicks'),
         Input('tabs', 'value')]
    )
    def refresh_serial_ports(n_clicks, tab):
        """
        Detect and list available serial ports.
        """
        try:
            ports = serial.tools.list_ports.comports()
            options = [
                {'label': f'{port.device} - {port.description}', 'value': port.device}
                for port in ports
            ]

            if not options:
                info = html.Div([
                    html.Span("⚠️ No serial ports detected. ",
                              style={'color': '#856404'}),
                    html.Span("Please connect your device and click Refresh.")
                ])
                return [], info

            info = html.Div([
                html.Span(f"✅ Found {len(ports)} port(s). ",
                          style={'color': '#155724'}),
                html.Span("Select a port to continue.")
            ])

            return options, info

        except Exception as e:
            return [], html.Div(f"❌ Error detecting ports: {str(e)}", style={'color': '#721c24'})

    @app.callback(
        [Output('code-preview', 'value'),
         Output('code-generation-status', 'children'),
         Output('code-preview-section', 'style'),
         Output('generated-code-store', 'data'),
         Output('compile-btn', 'disabled'),
         Output('compile-flash-btn', 'disabled'),
         Output('compile-btn', 'style'),
         Output('compile-flash-btn', 'style')],
        Input('generate-code-btn', 'n_clicks'),
        [State('deployment-model-selector', 'value'),
         State('target-board-selector', 'value'),
         State('optimization-level', 'value'),
         State('deployment-stride', 'value'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def generate_embedded_code(n_clicks, model_filename, target_board, optimization, stride, base_dir):
        """
        Generate embedded C/C++ code from the trained model using actual metadata.
        Model type is automatically detected from the selected model.
        Parameters are loaded from model metadata to ensure consistency.
        Uses the working directory from the store.
        """
        if not model_filename:
            return no_update, html.Div("⚠️ Please select a model first",
                                       style={'color': '#ff9800', 'padding': '10px'}), {'display': 'none'}, {}, True, True, no_update, no_update

        try:
            # Use stored base directory or default to PERSISTENT_DIR
            if not base_dir:
                base_dir = PERSISTENT_DIR

            models_dir = os.path.join(base_dir, 'models')
            models_metadata_file = os.path.join(
                models_dir, 'trained_models.json')

            with open(models_metadata_file, 'r') as f:
                models_metadata = json.load(f)

            metadata = models_metadata.get(model_filename, {})
            model_type = metadata.get('model_type', 'unknown')
            features = metadata.get('features', 0)
            classes = metadata.get('classes', 0)

            # Get training parameters from metadata
            model_params = metadata.get('model_params', {})
            sampling_rate = model_params.get('sampling_rate', 100)
            window_size_ms = model_params.get('window_size_ms', 1500)
            window_size_samples = int((window_size_ms / 1000) * sampling_rate)

            # Convert overlap percentage to stride samples and fraction
            overlap_percent = stride if stride is not None else 0  # Default: 0% overlap
            overlap_percent = max(
                0, min(99, overlap_percent))  # Clamp to 0-99%
            # Convert to 0.0-0.99 range for generator
            overlap_fraction = overlap_percent / 100.0
            stride_percent = 100 - overlap_percent  # Convert overlap to stride
            stride_samples = int((stride_percent / 100.0)
                                 * window_size_samples)
            stride_samples = max(1, stride_samples)  # Ensure at least 1 sample

            # Load the trained model to get actual parameters
            model_path = get_model_path(model_filename, base_dir)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = EdgeMLModel.load_model(model_path)

            # Get feature names from model or training metadata
            feature_names = model.feature_names
            if not feature_names:
                # Try to get from training metadata
                training_dir = os.path.join(base_dir, 'training')
                metadata_files = glob.glob(
                    os.path.join(training_dir, '*_metadata.json'))
                if metadata_files:
                    with open(metadata_files[0], 'r') as f:
                        training_metadata = json.load(f)
                        feature_names = training_metadata.get(
                            'feature_names', [])

            if not feature_names:
                # Last resort: load from training CSV
                train_files = glob.glob(
                    os.path.join(training_dir, '*_train.csv'))
                if train_files:
                    import pandas as pd
                    temp_df = pd.read_csv(train_files[0])
                    feature_names = [
                        col for col in temp_df.columns if col != 'label']

            # Get class names
            classes = list(model.label_encoder.classes_) if model.label_encoder else [
                'activity_1', 'activity_2']

            # Map target board to platform string for code generator
            platform_mapping = {
                'arduino:avr:uno': 'arduino',
                'arduino:avr:nano': 'arduino',
                'esp32:esp32:esp32': 'esp32',
                'esp32:esp32:esp32s3': 'esp32',
                'm5stack:esp32:m5stick_c': 'esp32',
                'seeed:nrf52:xiaonRF52840Sense': 'seeed_xiao',
                'STM32:stm32:GenF4': 'arm_cortex_m'
            }
            platform = platform_mapping.get(target_board, 'arduino')

            # Infer feature_method if not present in metadata
            if 'feature_method' not in model_params and feature_names:
                try:
                    has_acc_mag = any(str(name).startswith('acc_mag_')
                                      for name in feature_names)
                    has_gyro_mag = any(str(name).startswith('gyro_mag_')
                                       for name in feature_names)
                    if has_acc_mag and has_gyro_mag:
                        model_params['feature_method'] = 'orientation_invariant'
                except Exception:
                    pass

            # Prepare model data for code generation
            model_data = {
                'model_type': model_type,
                'feature_names': feature_names or [],
                'classes': classes,
                'model_params': model_params,
                'model_object': model,  # Pass actual model for parameter extraction
                # Pass full metadata so base_generator can read feature_config
                'model_info': metadata
            }

            # Generate code using proper code generators
            generated_code_files = generate_deployment_code(
                model_type, model_data, platform, optimization, overlap_fraction
            )

            # Run pre-deployment validation
            # Map platform to device key for resource estimation
            device_key_mapping = {
                'arduino': 'arduino_uno',
                'esp32': 'esp32',
                'seeed_xiao': 'seeed_xiao_nrf52840',
                'arm_cortex_m': 'stm32f4',
            }
            device_key = device_key_mapping.get(platform)
            
            # Add optimization and extracted model params for validation
            validation_model_data = model_data.copy()
            validation_model_data['optimization'] = optimization
            
            validation_report = validate_before_deployment(
                validation_model_data, generated_code_files, device_key
            )

            # Also save to working directory in organized structure
            output_dir = os.path.join(base_dir, 'generated')
            saved_files = generate_and_save_deployment_code(
                model_type, model_data, platform, output_dir, optimization, overlap_fraction
            )

            # Get the first generated file for preview (typically the sketch/example)
            # Priority: sketch > source > header
            sketch_file = None
            source_file = None
            header_file = None

            for filename, code in generated_code_files.items():
                if '.ino' in filename or 'example' in filename.lower():
                    sketch_file = (filename, code)
                elif '.cpp' in filename or '.c' in filename:
                    source_file = (filename, code)
                elif '.h' in filename:
                    header_file = (filename, code)

            # Show sketch first, then source, then header
            preview_file = sketch_file or source_file or header_file or list(
                generated_code_files.items())[0]
            preview_code = preview_file[1]
            preview_filename = preview_file[0]

            # Build validation summary for display
            validation_items = []
            code_checks = validation_report.get('checks', {}).get('code', {})
            if code_checks:
                for check in code_checks.get('checks_passed', []):
                    validation_items.append(html.Li(f"✅ {check}", style={'color': '#155724'}))
                for issue in code_checks.get('issues', []):
                    validation_items.append(html.Li(f"❌ {issue}", style={'color': '#721c24', 'font-weight': 'bold'}))
                for warning in code_checks.get('warnings', []):
                    validation_items.append(html.Li(f"⚠️ {warning}", style={'color': '#856404'}))
            
            validation_passed = validation_report.get('passed', True)
            validation_div = html.Div([
                html.Strong(
                    "🔍 Pre-deployment Validation: " + 
                    ("PASSED ✅" if validation_passed else "ISSUES FOUND ❌"),
                    style={'color': '#155724' if validation_passed else '#721c24'}
                ),
                html.Ul(validation_items, style={'margin-top': '5px', 'font-size': '12px'})
                if validation_items else None
            ], style={
                'margin-top': '10px', 'padding': '10px',
                'background': '#d4edda' if validation_passed else '#f8d7da',
                'border-radius': '4px', 'font-size': '13px'
            })

            status = html.Div([
                html.H5("✅ Code Generated Successfully!",
                        style={'color': '#28a745'}),
                html.P(
                    f"Model Type: {model_type.upper()} | Platform: {platform} | Optimization: {optimization.upper()}"),
                validation_div,
                html.Div([
                    html.Strong("📁 Generated Files: "),
                    html.Ul([
                        html.Li(filename, style={'font-family': 'monospace'})
                        for filename in generated_code_files.keys()
                    ])
                ], style={'margin-top': '10px', 'padding': '10px', 'background': '#e8f5e9', 'border-radius': '4px', 'font-size': '13px'}),
                html.Div([
                    html.Strong("💾 Saved to: "),
                    html.Code(output_dir, style={
                              'background': '#f8f9fa', 'padding': '2px 8px', 'border-radius': '3px'}),
                    html.Ul([
                        html.Li(os.path.relpath(filepath, base_dir), style={
                                'font-family': 'monospace', 'font-size': '12px'})
                        for filepath in saved_files.keys()
                    ], style={'margin-top': '5px'})
                ], style={'margin-top': '10px', 'padding': '10px', 'background': '#d1ecf1', 'border-radius': '4px', 'font-size': '13px'}),
                html.Div([
                    html.Strong("⚠️ Note: "),
                    html.Span(
                        f"Showing preview of {preview_filename}. All files will be included in download.")
                ], style={'margin-top': '10px', 'padding': '10px', 'background': '#fff3cd', 'border-radius': '4px', 'font-size': '13px'}),
                html.Div([
                    html.Span(f"Model uses: {sampling_rate} Hz, {window_size_ms} ms window, {len(feature_names)} features", style={
                              'font-size': '12px', 'color': '#666'})
                ], style={'margin-top': '8px'})
            ], style={'background': '#d4edda', 'padding': '15px', 'border-radius': '5px'})

            # Enable compile buttons
            compile_btn_style = {
                'background-color': '#007bff',
                'color': 'white',
                'border': 'none',
                'padding': '12px 25px',
                'border-radius': '6px',
                'cursor': 'pointer',
                'font-weight': 'bold',
                'margin-right': '10px',
                'opacity': '1'
            }

            flash_btn_style = {
                'background-color': '#dc3545',
                'color': 'white',
                'border': 'none',
                'padding': '12px 25px',
                'border-radius': '6px',
                'cursor': 'pointer',
                'font-weight': 'bold',
                'opacity': '1'
            }

            # Get board name for display
            board_names = {
                'arduino:avr:uno': 'Arduino Uno',
                'arduino:avr:nano': 'Arduino Nano',
                'esp32:esp32:esp32': 'ESP32',
                'esp32:esp32:esp32s3': 'ESP32-S3',
                'm5stack:esp32:m5stick_c': 'M5StickC Plus2',
                'seeed:nrf52:xiaonRF52840Sense': 'XIAO nRF52840 Sense',
                'STM32:stm32:GenF4': 'STM32F4'
            }
            board_name = board_names.get(target_board, 'Unknown Board')

            # Generate PlatformIO configuration
            platformio_ini = generate_platformio_config(
                target_board, model_filename, board_name)

            # Store all generated files
            code_data = {
                'code': preview_code,  # Main code for compilation
                'filename': preview_filename,
                'board': target_board,
                'platformio_ini': platformio_ini,
                'all_files': generated_code_files  # Store all generated files
            }

            return preview_code, status, {'display': 'block'}, code_data, False, False, compile_btn_style, flash_btn_style

        except Exception as e:
            print(f"Error generating code: {e}")
            import traceback
            traceback.print_exc()
            error_msg = html.Div([
                html.H5("❌ Code Generation Failed",
                        style={'color': '#dc3545'}),
                html.P(str(e))
            ], style={'background': '#f8d7da', 'padding': '15px', 'border-radius': '5px'})
            return no_update, error_msg, {'display': 'none'}, {}, True, True, no_update, no_update

    @app.callback(
        Output('download-generated-code', 'data'),
        Input('download-code-btn', 'n_clicks'),
        State('generated-code-store', 'data'),
        prevent_initial_call=True
    )
    def download_code(n_clicks, code_data):
        """
        Download generated code as .ino file.
        """
        if not code_data or 'code' not in code_data:
            return no_update

        from dash import dcc
        return dcc.send_string(code_data['code'], filename=code_data['filename'])

    @app.callback(
        [Output('compilation-output', 'children'),
         Output('compilation-section', 'style'),
         Output('flash-progress', 'children')],
        [Input('compile-btn', 'n_clicks'),
         Input('compile-flash-btn', 'n_clicks')],
        [State('generated-code-store', 'data'),
         State('serial-port-selector-deploy', 'value'),
         State('target-board-selector', 'value'),
         State('toolchain-selector', 'value'),
         State('compilation-options', 'value')],
        prevent_initial_call=True
    )
    def compile_and_flash(compile_clicks, flash_clicks, code_data, serial_port, board_fqbn, toolchain, options):
        """
        Compile code using Arduino CLI or PlatformIO and optionally flash to device.
        """
        if not code_data or 'code' not in code_data:
            return "No code to compile", {'display': 'block'}, ""

        try:
            # Determine which button was clicked
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            should_upload = button_id == 'compile-flash-btn'

            if should_upload and not serial_port:
                return "❌ Please select a serial port for flashing", {'display': 'block'}, ""

            verbose = 'verbose' in options

            # Create temporary project directory
            import tempfile
            import shutil
            temp_dir = tempfile.mkdtemp(prefix='har_build_')

            try:
                if toolchain == 'arduino':
                    output, success = compile_with_arduino_cli(
                        code_data, temp_dir, board_fqbn, serial_port, should_upload, verbose)
                else:  # platformio
                    output, success = compile_with_platformio(
                        code_data, temp_dir, serial_port, should_upload, verbose)
            finally:
                # Cleanup temp directory
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

            if not success:
                return output, {'display': 'block'}, ""

            progress = ""
            if should_upload:
                progress = html.Div([
                    html.Div("Upload Progress:", style={
                             'font-weight': 'bold', 'margin-bottom': '10px'}),
                    html.Div([
                        html.Div(style={
                            'width': '100%',
                            'height': '30px',
                            'background': '#28a745',
                            'border-radius': '15px',
                            'position': 'relative'
                        }),
                        html.Div("✅ Upload Complete!", style={
                            'text-align': 'center',
                            'margin-top': '10px',
                            'color': '#28a745',
                            'font-weight': 'bold'
                        })
                    ])
                ], style={'background': '#d4edda', 'padding': '15px', 'border-radius': '5px'})

            return output, {'display': 'block'}, progress

        except Exception as e:
            error_output = f"❌ Error during compilation:\n\n{str(e)}\n\n{traceback.format_exc()}"
            return error_output, {'display': 'block'}, ""

    @app.callback(
        Output('code-preview', 'value', allow_duplicate=True),
        Input('copy-code-btn', 'n_clicks'),
        State('code-preview', 'value'),
        prevent_initial_call=True
    )
    def copy_to_clipboard(n_clicks, code):
        """
        Copy code to clipboard (handled by browser).
        """
        # Note: Actual clipboard copying requires clientside callback or JS
        return code

    @app.callback(
        Output('resource-analysis-output', 'children'),
        Input('resource-analysis-btn', 'n_clicks'),
        [State('deployment-model-selector', 'value'),
         State('target-board-selector', 'value'),
         State('optimization-level', 'value'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def analyze_resources(n_clicks, model_filename, target_board, optimization, base_dir):
        """
        Analyze and display resource requirements for deploying the model to target board.
        Uses the working directory from the store.
        """
        if not model_filename:
            return html.Div("⚠️ Please select a model first", style={'color': '#ff9800', 'padding': '10px'})

        try:
            # Use stored base directory or default to PERSISTENT_DIR
            if not base_dir:
                base_dir = PERSISTENT_DIR

            models_dir = os.path.join(base_dir, 'models')
            models_metadata_file = os.path.join(
                models_dir, 'trained_models.json')

            with open(models_metadata_file, 'r') as f:
                models_metadata = json.load(f)

            metadata = models_metadata.get(model_filename, {})
            model_type = metadata.get('model_type', 'unknown')
            features = metadata.get('features', 0)
            classes = metadata.get('classes', 0)

            # Get model params
            model_params = metadata.get('model_params', {})
            sampling_rate = model_params.get('sampling_rate', 100)
            window_size_ms = model_params.get('window_size_ms', 1500)
            window_size_samples = int((window_size_ms / 1000) * sampling_rate)

            # Estimate resource usage based on model type and target board
            board_specs = {
                'arduino:avr:uno': {'name': 'Arduino Uno', 'ram': 2, 'flash': 32, 'speed': 16},
                'arduino:avr:nano': {'name': 'Arduino Nano', 'ram': 2, 'flash': 32, 'speed': 16},
                'esp32:esp32:esp32': {'name': 'ESP32', 'ram': 520, 'flash': 4096, 'speed': 240},
                'esp32:esp32:esp32s3': {'name': 'ESP32-S3', 'ram': 512, 'flash': 8192, 'speed': 240},
                'm5stack:esp32:m5stick_c': {'name': 'M5StickC Plus2', 'ram': 320, 'flash': 8192, 'speed': 240},
                'seeed:nrf52:xiaonRF52840Sense': {'name': 'XIAO nRF52840 Sense', 'ram': 256, 'flash': 1024, 'speed': 64},
                'STM32:stm32:GenF4': {'name': 'STM32F4', 'ram': 192, 'flash': 1024, 'speed': 168}
            }

            board_info = board_specs.get(
                target_board, {'name': 'Unknown', 'ram': 100, 'flash': 1024, 'speed': 100})

            # Estimate RAM usage (buffer + feature array + model weights)
            sensor_buffer_kb = (window_size_samples * 6 * 4) / \
                1024  # 6 sensors, 4 bytes per float
            feature_buffer_kb = (features * 4) / 1024  # 4 bytes per float

            # Model weights estimation
            if model_type == 'neural_network':
                # Assume 2 hidden layers with 64, 32 neurons
                weights_kb = (
                    (features * 64 + 64 * 32 + 32 * classes) * 4) / 1024
            elif model_type == 'svm':
                weights_kb = (features * classes * 4) / 1024
            else:
                weights_kb = (features * 2 * 4) / 1024  # Generic estimate

            total_ram_kb = sensor_buffer_kb + feature_buffer_kb + \
                weights_kb + 10  # +10 for stack/heap
            ram_usage_percent = (total_ram_kb / board_info['ram']) * 100

            # Estimate Flash usage
            code_size_kb = 50 + weights_kb  # 50KB for code + model weights
            flash_usage_percent = (code_size_kb / board_info['flash']) * 100

            # Estimate inference time (based on optimization level)
            opt_multipliers = {'accuracy': 1.5,
                               'balanced': 1.0, 'speed': 0.7, 'power': 1.2}
            base_time_ms = (features * 0.1) + \
                (classes * 0.05)  # Rough estimate
            inference_time_ms = base_time_ms * \
                opt_multipliers.get(optimization, 1.0)

            # Power consumption estimate (mW)
            if 'esp32' in target_board or 'm5stack' in target_board:
                power_mw = 160 if optimization == 'power' else 240
            elif 'stm32' in target_board.lower():
                power_mw = 80 if optimization == 'power' else 120
            else:
                power_mw = 40 if optimization == 'power' else 60

            # Build resource analysis display
            return html.Div([
                html.H5("📊 Resource Analysis Results", style={
                        'color': '#28a745', 'margin-bottom': '15px'}),

                html.Div([
                    # Board Info
                    html.Div([
                        html.H6(f"🎯 Target: {board_info['name']}", style={
                                'color': '#2E86AB', 'margin-bottom': '10px'}),
                        html.Div(f"RAM: {board_info['ram']} KB | Flash: {board_info['flash']} KB | Clock: {board_info['speed']} MHz",
                                 style={'font-size': '12px', 'color': '#666'})
                    ], style={'margin-bottom': '20px'}),

                    # Resource Usage
                    html.Div([
                        # RAM Usage
                        html.Div([
                            html.Div([
                                html.Span("💾 RAM Usage: ", style={
                                          'font-weight': 'bold'}),
                                html.Span(f"{total_ram_kb:.1f} KB / {board_info['ram']} KB ({ram_usage_percent:.1f}%)",
                                          style={'color': '#dc3545' if ram_usage_percent > 80 else '#28a745' if ram_usage_percent < 50 else '#ff9800'})
                            ], style={'margin-bottom': '5px'}),
                            html.Div(style={
                                'width': '100%',
                                'height': '20px',
                                'background': '#e9ecef',
                                'border-radius': '10px',
                                'overflow': 'hidden'
                            }, children=[
                                html.Div(style={
                                    'width': f'{min(ram_usage_percent, 100):.1f}%',
                                    'height': '100%',
                                    'background': '#dc3545' if ram_usage_percent > 80 else '#28a745' if ram_usage_percent < 50 else '#ff9800',
                                    'transition': 'width 0.3s'
                                })
                            ])
                        ], style={'margin-bottom': '15px'}),

                        # Flash Usage
                        html.Div([
                            html.Div([
                                html.Span("💿 Flash Usage: ", style={
                                          'font-weight': 'bold'}),
                                html.Span(f"{code_size_kb:.1f} KB / {board_info['flash']} KB ({flash_usage_percent:.1f}%)",
                                          style={'color': '#dc3545' if flash_usage_percent > 80 else '#28a745'})
                            ], style={'margin-bottom': '5px'}),
                            html.Div(style={
                                'width': '100%',
                                'height': '20px',
                                'background': '#e9ecef',
                                'border-radius': '10px',
                                'overflow': 'hidden'
                            }, children=[
                                html.Div(style={
                                    'width': f'{min(flash_usage_percent, 100):.1f}%',
                                    'height': '100%',
                                    'background': '#dc3545' if flash_usage_percent > 80 else '#28a745',
                                    'transition': 'width 0.3s'
                                })
                            ])
                        ], style={'margin-bottom': '15px'}),

                        # Performance Metrics
                        html.Div([
                            html.Div([
                                html.Span("⚡ Inference Time: ", style={
                                          'font-weight': 'bold'}),
                                html.Span(f"{inference_time_ms:.2f} ms",
                                          style={'color': '#2E86AB'}),
                                html.Span(f" (~{1000/inference_time_ms:.1f} Hz max)", style={
                                          'font-size': '11px', 'color': '#999', 'margin-left': '5px'})
                            ], style={'margin-bottom': '8px'}),
                            html.Div([
                                html.Span("🔋 Power Consumption: ",
                                          style={'font-weight': 'bold'}),
                                html.Span(f"{power_mw} mW", style={
                                          'color': '#2E86AB'}),
                                html.Span(f" ({optimization} mode)", style={
                                          'font-size': '11px', 'color': '#999', 'margin-left': '5px'})
                            ], style={'margin-bottom': '8px'}),
                            html.Div([
                                html.Span("🎯 Optimization: ", style={
                                          'font-weight': 'bold'}),
                                html.Span(f"{optimization.upper()}",
                                          style={'color': '#2E86AB'})
                            ])
                        ])
                    ], style={'padding': '15px', 'background': '#f8f9fa', 'border-radius': '8px'}),

                    # Warnings
                    html.Div([
                        html.Div("⚠️ Warnings:", style={'font-weight': 'bold', 'margin-bottom': '10px',
                                 'color': '#856404'}) if ram_usage_percent > 80 or flash_usage_percent > 80 else None,
                        html.Div("• RAM usage exceeds 80% - consider reducing window size or features", style={
                                 'color': '#721c24', 'margin-bottom': '5px'}) if ram_usage_percent > 80 else None,
                        html.Div("• Flash usage exceeds 80% - consider model optimization",
                                 style={'color': '#721c24'}) if flash_usage_percent > 80 else None
                    ], style={'margin-top': '15px', 'padding': '12px', 'background': '#fff3cd', 'border-radius': '6px', 'border-left': '4px solid #ffc107'}) if ram_usage_percent > 80 or flash_usage_percent > 80 else None
                ])
            ], style={'background': '#d4edda', 'padding': '20px', 'border-radius': '8px', 'border-left': '4px solid #28a745'})

        except Exception as e:
            print(f"Error analyzing resources: {e}")
            import traceback
            traceback.print_exc()
            return html.Div([
                html.H5("❌ Resource Analysis Failed",
                        style={'color': '#dc3545'}),
                html.P(str(e))
            ], style={'background': '#f8d7da', 'padding': '15px', 'border-radius': '5px'})
