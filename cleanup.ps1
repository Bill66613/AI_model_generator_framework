# Cleanup Script for HAR Framework
# This script moves development/testing files to an archive folder

Write-Host "HAR Framework Cleanup Script" -ForegroundColor Cyan
Write-Host "============================`n" -ForegroundColor Cyan

# Create archive directory
$archiveDir = "archive_dev_files"
if (-not (Test-Path $archiveDir)) {
    New-Item -ItemType Directory -Path $archiveDir | Out-Null
    Write-Host "Created archive directory: $archiveDir" -ForegroundColor Green
}

# Function to move files safely
function Move-FileToArchive {
    param($fileName)
    if (Test-Path $fileName) {
        Move-Item $fileName $archiveDir -Force
        Write-Host "  Moved: $fileName" -ForegroundColor Yellow
        return 1
    }
    return 0
}

# Count moved files
$movedCount = 0

# Move test files (except test_serial_output.py - that's useful for users)
Write-Host "`nMoving test files..." -ForegroundColor Cyan
$movedCount += Move-FileToArchive "test_callback_integration.py"
$movedCount += Move-FileToArchive "test_deployment.py"
$movedCount += Move-FileToArchive "test_fixed_generator.py"
$movedCount += Move-FileToArchive "test_model_predictions.py"
$movedCount += Move-FileToArchive "test_multi_model_organized.py"
$movedCount += Move-FileToArchive "test_organized_generation.py"
$movedCount += Move-FileToArchive "test_refactored_generators.py"
$movedCount += Move-FileToArchive "test_svm_generator.py"
$movedCount += Move-FileToArchive "test_svm_integration.py"
$movedCount += Move-FileToArchive "test.py"

# Move diagnostic scripts
Write-Host "`nMoving diagnostic scripts..." -ForegroundColor Cyan
$movedCount += Move-FileToArchive "analyze_motion.py"
$movedCount += Move-FileToArchive "check_class_distribution.py"
$movedCount += Move-FileToArchive "check_feature_mismatch.py"
$movedCount += Move-FileToArchive "check_labels.py"
$movedCount += Move-FileToArchive "check_model_features.py"
$movedCount += Move-FileToArchive "check_model_structure.py"
$movedCount += Move-FileToArchive "check_nn_model.py"
$movedCount += Move-FileToArchive "check_normalization.py"
$movedCount += Move-FileToArchive "compare_feature_extraction.py"
$movedCount += Move-FileToArchive "compute_correct_scaler.py"
$movedCount += Move-FileToArchive "diagnose_model.py"
$movedCount += Move-FileToArchive "extract_final_layer.py"
$movedCount += Move-FileToArchive "show_all_features.py"
$movedCount += Move-FileToArchive "verify_fix.py"
$movedCount += Move-FileToArchive "verify_labels.py"
$movedCount += Move-FileToArchive "verify_project.py"

# Move migration scripts
Write-Host "`nMoving migration scripts..." -ForegroundColor Cyan
$movedCount += Move-FileToArchive "migrate_storage.py"
$movedCount += Move-FileToArchive "regenerate_deployment.py"
$movedCount += Move-FileToArchive "retrain_without_frequency.py"
$movedCount += Move-FileToArchive "convert_uci_har_to_csv.py"
$movedCount += Move-FileToArchive "split_uci_har_files.py"
$movedCount += Move-FileToArchive "organized_generation_examples.py"

# Move redundant markdown files
Write-Host "`nMoving redundant documentation..." -ForegroundColor Cyan
$movedCount += Move-FileToArchive "3WAY_SPLIT_IMPLEMENTATION.md"
$movedCount += Move-FileToArchive "CLASS_IMBALANCE_FIX.md"
$movedCount += Move-FileToArchive "DEPLOYMENT_READY.md"
$movedCount += Move-FileToArchive "DIRECTORY_AUDIT.md"
$movedCount += Move-FileToArchive "FEATURE_EXTRACTION_ANALYSIS.md"
$movedCount += Move-FileToArchive "FEATURE_ORDER_BUG_FIX.md"
$movedCount += Move-FileToArchive "FEATURE_TERMINOLOGY_FIX.md"
$movedCount += Move-FileToArchive "FRAMEWORK_RESTRUCTURE_COMPLETE.md"
$movedCount += Move-FileToArchive "NEURAL_NETWORK_GENERATOR_FIX.md"
$movedCount += Move-FileToArchive "ORGANIZED_GENERATION_README.md"
$movedCount += Move-FileToArchive "SCALER_BUG_FIX.md"
$movedCount += Move-FileToArchive "SLIDING_WINDOW_IMPLEMENTATION.md"
$movedCount += Move-FileToArchive "STORAGE_PATH_FIX.md"
$movedCount += Move-FileToArchive "STORAGE_RESTRUCTURE_SUMMARY.md"
$movedCount += Move-FileToArchive "SVM_GENERATOR_FIXES.md"
$movedCount += Move-FileToArchive "THRESHOLD_FIX_GUIDE.md"
$movedCount += Move-FileToArchive "TRAINING_IMPROVEMENTS.md"
$movedCount += Move-FileToArchive "UI_INTEGRATION_SUMMARY.md"
$movedCount += Move-FileToArchive "WORKING_DIRECTORY_FEATURE.md"

# Summary
Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "Cleanup Summary" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Files moved to archive: $movedCount" -ForegroundColor Green

Write-Host "`nKept files (important):" -ForegroundColor Cyan
Write-Host "  - README.md" -ForegroundColor Green
Write-Host "  - FRAMEWORK_DOCUMENTATION.md" -ForegroundColor Green
Write-Host "  - DEVELOPMENT_HISTORY.md" -ForegroundColor Green
Write-Host "  - QUICK_REFERENCE.md" -ForegroundColor Green
Write-Host "  - TODO.md" -ForegroundColor Green
Write-Host "  - DATA_COLLECTION_GUIDE.md" -ForegroundColor Green
Write-Host "  - test_serial_output.py (user tool)" -ForegroundColor Green
Write-Host "  - requirements.txt" -ForegroundColor Green

Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "  1. Review files in $archiveDir" -ForegroundColor Yellow
Write-Host "  2. If satisfied, compress: Compress-Archive -Path $archiveDir -DestinationPath ${archiveDir}.zip" -ForegroundColor Yellow
Write-Host "  3. Or delete: Remove-Item $archiveDir -Recurse -Force" -ForegroundColor Yellow
Write-Host "  4. Update README.md to reference FRAMEWORK_DOCUMENTATION.md" -ForegroundColor Yellow

Write-Host "`nCleanup complete!`n" -ForegroundColor Green
