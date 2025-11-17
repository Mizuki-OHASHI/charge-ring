init:
	chmod u+x scripts/*.sh
job:
	scripts/job.sh
s12n:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251027_215518.npz --exp_vtip data/exp_data.csv --exp_htip data/exp_data_z.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 8
s12p:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251027_215518.npz --exp_vtip data/exp_data_ringA_Vtip+.csv --exp_htip data/exp_data_z_ringA_Vtip+.csv --cpd $(CPD) --fixed_vtip 4 --fixed_htip 8
s11n:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251116_213253.npz --exp_vtip data/exp_data.csv --exp_htip data/exp_data_z.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 8
s11p:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251116_213253.npz --exp_vtip data/exp_data_ringA_Vtip+.csv --exp_htip data/exp_data_z_ringA_Vtip+.csv --cpd $(CPD) --fixed_vtip 4 --fixed_htip 8
s10n:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251117_173915.npz --exp_vtip data/exp_data.csv --exp_htip data/exp_data_z.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 8
s10p:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251117_173915.npz --exp_vtip data/exp_data_ringA_Vtip+.csv --exp_htip data/exp_data_z_ringA_Vtip+.csv --cpd $(CPD) --fixed_vtip 4 --fixed_htip 8