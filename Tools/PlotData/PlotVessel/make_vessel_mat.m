% make_vessel_mat.m
%
% Extract TCV vessel and tile geometry from MDS+ for a given shot and save
% it as a .mat file compatible with plotVessel.py.
%
% USAGE:
%   shot = 88612;
%   make_vessel_mat(shot);
%   % Saves: <shot>_vessel.mat in the current directory.
%
% The geometry stored is shot-dependent because MDS+ 'static' tree snapshots
% can change between machine configurations.

function make_vessel_mat(shot)

    fprintf('Opening static tree for shot %d ...\n', shot);
    mdsopen('static', shot);

    % --- Vessel inner/outer wall (piecewise-polynomial, evaluated on 255 pts) --
    ppv_i = ppmak(mdsvalue('static(''pp_v:in:breaks'')')', ...
                  mdsvalue('static(''pp_v:in:coefs'')'), 2);
    ppv_o = ppmak(mdsvalue('static(''pp_v:out:breaks'')')', ...
                  mdsvalue('static(''pp_v:out:coefs'')'), 2);

    ss = linspace(0, 1, 255);
    RZv_i = ppual(ppv_i, ss);   % (2, 255)  [R; Z] in metres
    RZv_o = ppual(ppv_o, ss);

    % --- Tile aperture contour (closed, CW from inboard midplane) --------------
    Rt = mdsdata('static(''r_tv'')');
    Zt = mdsdata('static(''z_tv'')');

    mdsclose();

    % --- Pack into struct matching plotVessel.py expectations ------------------
    vesselcont.Rv_in  = RZv_i(1, :);   % (1, 255)
    vesselcont.Zv_in  = RZv_i(2, :);
    vesselcont.Rv_out = RZv_o(1, :);
    vesselcont.Zv_out = RZv_o(2, :);
    vesselcont.Rt     = Rt(:);          % column vector
    vesselcont.Zt     = Zt(:);

    filename = sprintf('%d_vessel.mat', shot);
    save(filename, 'vesselcont');
    fprintf('Saved: %s\n', filename);

end
