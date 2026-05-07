% generate_scope_reference.m
%
% Regenerate Helios's frozen SCOPE v2.0 reference CSV for the SIF Tier 2
% intercomparison test. See README.md in this directory for provenance and
% prerequisites.
%
% Usage (from within MATLAB, with SCOPE cloned to ~/Downloads/SCOPE):
%
%   cd('/path/to/Helios/plugins/radiation/tests/reference');
%   SCOPE_DIR = '~/Downloads/SCOPE';       % edit if SCOPE is elsewhere
%   generate_scope_reference(SCOPE_DIR);
%
% Or batched from a shell:
%
%   /Applications/MATLAB_R2026a.app/bin/matlab -nodisplay -nosplash -nodesktop \
%     -batch "generate_scope_reference('/Users/you/Downloads/SCOPE');"
%
% Prerequisites:
%   1. Clone SCOPE: git clone https://github.com/Christiaanvandertol/SCOPE.git
%   2. In SCOPE's set_parameter_filenames.csv, the line must read:
%        setoptions.csv, filenames.csv, input_data_default.csv
%   3. In SCOPE's input/setoptions.csv, set `0,verify` (single-point run).
%
% What this script does:
%   1. Runs SCOPE from its stock default input deck (LAI=3, Cab=40, SZA=30°, ...).
%   2. Locates the most recent SCOPE output directory.
%   3. Extracts top-of-canopy hemispheric and nadir fluorescence at 685 nm and
%      740 nm, plus band-integrated hemispheric SIF over 680-700 nm (SIF_red)
%      and 730-760 nm (SIF_farred).
%   4. Extracts leaf rho/tau at SIF bands by re-running fluspect_B_CX with the
%      default input leaf biochemistry.
%   5. Extracts soil reflectance at SIF bands from SCOPE's rsd.csv output.
%   6. Writes scope_v2_homogeneous_lai3.csv with the frozen reference values.
%
% IMPORTANT: After running this script, edit the CSV header to update:
%   - scope_git_commit  (from `git -C <SCOPE_DIR> log -1 --format=%h`)
%   - matlab_version    (from `version('-release')`)
%   - run_date          (today's date)

function generate_scope_reference(scope_dir)

    if nargin < 1
        scope_dir = '~/Downloads/SCOPE';
    end
    scope_dir = char(scope_dir);
    if scope_dir(1) == '~'
        scope_dir = [getenv('HOME') scope_dir(2:end)];
    end
    fprintf('Using SCOPE at: %s\n', scope_dir);

    % ---- Run SCOPE from its top directory ----
    orig_pwd = pwd;
    cleanup = onCleanup(@() cd(orig_pwd));
    cd(scope_dir);
    addpath(genpath(pwd));
    fprintf('Running SCOPE...\n');
    SCOPE;

    % ---- Locate newest output dir ----
    d = dir(fullfile(scope_dir, 'output'));
    d = d([d.isdir]);
    d = d(~ismember({d.name}, {'.', '..', 'verificationdata'}));
    [~, order] = sort([d.datenum], 'descend');
    newest = fullfile(scope_dir, 'output', d(order(1)).name);
    fprintf('Reading outputs from: %s\n', newest);

    % ---- Load SCOPE fluorescence output grids ----
    wlF = load(fullfile(newest, 'wlF.txt'));        % 640:1:850 nm
    wlF = wlF(1, :);                                 % in case multiple rows
    hemis = read_scope_csv(fullfile(newest, 'fluorescence_hemis.csv'));
    nadir = read_scope_csv(fullfile(newest, 'fluorescence.csv'));
    assert(length(hemis) == length(wlF), 'fluorescence_hemis length mismatch');
    assert(length(nadir) == length(wlF), 'fluorescence length mismatch');

    idx_685 = find(wlF == 685, 1);
    idx_740 = find(wlF == 740, 1);
    sif_hemis_685 = hemis(idx_685);
    sif_hemis_740 = hemis(idx_740);
    sif_nadir_685 = nadir(idx_685);
    sif_nadir_740 = nadir(idx_740);

    mask_red    = (wlF >= 680 & wlF <= 700);
    mask_farred = (wlF >= 730 & wlF <= 760);
    % Values are W/m^2/um. Sum with 1 nm spacing and convert nm->um (×0.001).
    sif_hemis_band_red    = sum(hemis(mask_red))    * 0.001;
    sif_hemis_band_farred = sum(hemis(mask_farred)) * 0.001;

    % ---- Pre-reabsorption emitted SIF per band (for escape-fraction computation) ----
    % fluorescence_ReabsCorr.csv is nadir radiance of produced SIF (no reabsorption).
    % For a Lambertian source hemispheric flux = pi * nadir radiance.
    rc = read_scope_csv(fullfile(newest, 'fluorescence_ReabsCorr.csv'));
    sif_emitted_band_red    = sum(rc(mask_red))    * 0.001 * pi;
    sif_emitted_band_farred = sum(rc(mask_farred)) * 0.001 * pi;
    f_esc_band_red    = sif_hemis_band_red    / sif_emitted_band_red;
    f_esc_band_farred = sif_hemis_band_farred / sif_emitted_band_farred;

    % ---- Extract soil rsd at SIF bands ----
    wlS = load(fullfile(newest, 'wlS.txt'));
    wlS = wlS(:)';
    rsd = read_scope_csv(fullfile(newest, 'rsd.csv'));
    idx_685_S = find(wlS == 685, 1);
    idx_740_S = find(wlS == 740, 1);
    soil_rho_685 = rsd(idx_685_S);
    soil_rho_740 = rsd(idx_740_S);
    mask_red_S    = (wlS >= 680 & wlS <= 700);
    mask_farred_S = (wlS >= 730 & wlS <= 760);
    soil_rho_band_red    = mean(rsd(mask_red_S));
    soil_rho_band_farred = mean(rsd(mask_farred_S));

    % ---- Extract leaf rho/tau at SIF bands by running fluspect_B_CX ----
    spectral = define_bands;
    optipar = load('input/fluspect_parameters/Optipar2021_ProspectPRO_CX.mat');
    leafbio = struct('Cab',40,'Cca',10,'Cw',0.009,'Cdm',0.012,'Cs',0, ...
                     'Cant',1,'Cp',0,'Cbc',0,'N',1.5, ...
                     'rho_thermal',0.01,'tau_thermal',0.01,'fqe',0.01,'V2Z',0);
    leafopt = fluspect_B_CX(spectral, leafbio, optipar.optipar);
    wlP = spectral.wlP;
    idx_685_P = find(wlP == 685, 1);
    idx_740_P = find(wlP == 740, 1);
    leaf_rho_685 = leafopt.refl(idx_685_P);
    leaf_tau_685 = leafopt.tran(idx_685_P);
    leaf_rho_740 = leafopt.refl(idx_740_P);
    leaf_tau_740 = leafopt.tran(idx_740_P);
    mask_red_P    = (wlP >= 680 & wlP <= 700);
    mask_farred_P = (wlP >= 730 & wlP <= 760);
    leaf_rho_band_red    = mean(leafopt.refl(mask_red_P));
    leaf_tau_band_red    = mean(leafopt.tran(mask_red_P));
    leaf_rho_band_farred = mean(leafopt.refl(mask_farred_P));
    leaf_tau_band_farred = mean(leafopt.tran(mask_farred_P));

    % ---- Write the reference CSV ----
    cd(orig_pwd);
    out_path = 'scope_v2_homogeneous_lai3.csv';
    fid = fopen(out_path, 'w');
    fprintf(fid, '# Frozen SCOPE v2.0 reference output for Helios SIF Tier 2 intercomparison.\n');
    fprintf(fid, '# Provenance and regeneration instructions in this directory''s README.md.\n');
    fprintf(fid, '# Do NOT hand-edit values - regenerate using generate_scope_reference.m.\n');
    fprintf(fid, '# Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    fprintf(fid, 'key,value,units,note\n');
    wrow = @(k,v,u,n) fprintf(fid, '%s,%s,%s,%s\n', k, num2str(v), u, n);

    wrow('scope_git_commit','UPDATE_ME','','SCOPE commit used to generate these values');
    wrow('matlab_version',version('-release'),'','MATLAB used to run SCOPE');
    wrow('run_date',datestr(now,'yyyy-mm-dd'),'','YYYY-MM-DD');
    fprintf(fid, '# --- Scene parameters fed to SCOPE (input_data_default.csv) ---\n');
    wrow('LAI',3.0,'','leaf area index');
    wrow('Cab',40.0,'ug/cm^2','chlorophyll a+b content');
    wrow('N_prospect',1.5,'','PROSPECT mesophyll parameter');
    wrow('LIDFa',-0.35,'','leaf-inclination distribution function parameter a');
    wrow('LIDFb',-0.15,'','leaf-inclination distribution function parameter b');
    wrow('SZA',30.0,'deg','solar zenith angle (tts)');
    wrow('VZA',0.0,'deg','viewing zenith angle (tto)');
    wrow('Rin',600.0,'W/m^2','broadband incoming shortwave at top of canopy');
    wrow('Ta',20.0,'degC','air temperature');
    wrow('fqe',0.01,'','SCOPE intrinsic fluorescence quantum efficiency');
    fprintf(fid, '# --- Leaf optics at SIF band centres (from fluspect_B_CX) ---\n');
    wrow('leaf_rho_685', leaf_rho_685, '', 'leaf reflectance at 685 nm');
    wrow('leaf_tau_685', leaf_tau_685, '', 'leaf transmittance at 685 nm');
    wrow('leaf_rho_740', leaf_rho_740, '', 'leaf reflectance at 740 nm');
    wrow('leaf_tau_740', leaf_tau_740, '', 'leaf transmittance at 740 nm');
    wrow('leaf_rho_band_red',    leaf_rho_band_red,    '', 'leaf reflectance averaged over 680-700 nm');
    wrow('leaf_tau_band_red',    leaf_tau_band_red,    '', 'leaf transmittance averaged over 680-700 nm');
    wrow('leaf_rho_band_farred', leaf_rho_band_farred, '', 'leaf reflectance averaged over 730-760 nm');
    wrow('leaf_tau_band_farred', leaf_tau_band_farred, '', 'leaf transmittance averaged over 730-760 nm');
    fprintf(fid, '# --- Soil directional-hemispherical reflectance (rsd.csv) ---\n');
    wrow('soil_rho_685',          soil_rho_685,          '', 'soil reflectance at 685 nm');
    wrow('soil_rho_740',          soil_rho_740,          '', 'soil reflectance at 740 nm');
    wrow('soil_rho_band_red',     soil_rho_band_red,     '', 'soil reflectance averaged over 680-700 nm');
    wrow('soil_rho_band_farred',  soil_rho_band_farred,  '', 'soil reflectance averaged over 730-760 nm');
    fprintf(fid, '# --- SCOPE top-of-canopy fluorescence outputs ---\n');
    wrow('sif_hemis_685', sif_hemis_685, 'W/m^2/um',     'hemispheric TOC fluorescence at 685 nm');
    wrow('sif_hemis_740', sif_hemis_740, 'W/m^2/um',     'hemispheric TOC fluorescence at 740 nm');
    wrow('sif_nadir_685', sif_nadir_685, 'W/m^2/um/sr',  'nadir TOC fluorescence radiance at 685 nm');
    wrow('sif_nadir_740', sif_nadir_740, 'W/m^2/um/sr',  'nadir TOC fluorescence radiance at 740 nm');
    wrow('sif_hemis_band_red',    sif_hemis_band_red,    'W/m^2', 'hemispheric TOC fluorescence integrated over 680-700 nm');
    wrow('sif_hemis_band_farred', sif_hemis_band_farred, 'W/m^2', 'hemispheric TOC fluorescence integrated over 730-760 nm');
    fprintf(fid, '# --- Pre-reabsorption emitted SIF per band ---\n');
    wrow('sif_emitted_band_red',    sif_emitted_band_red,    'W/m^2', 'total SIF produced in 680-700 nm per m^2 ground');
    wrow('sif_emitted_band_farred', sif_emitted_band_farred, 'W/m^2', 'total SIF produced in 730-760 nm per m^2 ground');
    fprintf(fid, '# --- Canopy escape probability per band ---\n');
    wrow('f_esc_band_red',    f_esc_band_red,    '', 'escape fraction = hemis/emitted for SIF_red band');
    wrow('f_esc_band_farred', f_esc_band_farred, '', 'escape fraction = hemis/emitted for SIF_farred band');
    fclose(fid);
    fprintf('Wrote %s\n', out_path);
    fprintf('REMEMBER to update scope_git_commit at the top of the CSV.\n');
end

function vals = read_scope_csv(path)
    % Read a SCOPE output CSV that has comment lines starting with '#' and one
    % or more numeric rows. Returns the first numeric row as a row vector.
    fid = fopen(path, 'r');
    line = '';
    while true
        line = fgetl(fid);
        if ~ischar(line), fclose(fid); error('No data row in %s', path); end
        stripped = strtrim(line);
        if ~isempty(stripped) && stripped(1) ~= '#', break; end
    end
    fclose(fid);
    vals = sscanf(line, '%f,');
    if isempty(vals)
        vals = sscanf(line, '%f');  % whitespace-separated fallback
    end
    vals = vals(:)';
end
