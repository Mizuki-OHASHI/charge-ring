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
s3e12n:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251117_155024.npz --exp_vtip data/exp_data.csv --exp_htip data/exp_data_z.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 8
s3e12p:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251117_155024.npz --exp_vtip data/exp_data_ringA_Vtip+.csv --exp_htip data/exp_data_z_ringA_Vtip+.csv --cpd $(CPD) --fixed_vtip 4 --fixed_htip 8
s2e12n:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251118_012435.npz --exp_vtip data/exp_data.csv --exp_htip data/exp_data_z.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 8
s2e12p:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251118_012435.npz --exp_vtip data/exp_data_ringA_Vtip+.csv --exp_htip data/exp_data_z_ringA_Vtip+.csv --cpd $(CPD) --fixed_vtip 4 --fixed_htip 8
s9n:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251118_105446.npz --exp_vtip data/exp_data.csv --exp_htip data/exp_data_z.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 8
s9p:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251118_105446.npz --exp_vtip data/exp_data_ringA_Vtip+.csv --exp_htip data/exp_data_z_ringA_Vtip+.csv --cpd $(CPD) --fixed_vtip 4 --fixed_htip 8
s7e12n:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251118_182042.npz --exp_vtip data/exp_data.csv --exp_htip data/exp_data_z.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 8
s7e12p:
	python ring_radius_fitting.py --cache _ring_radius_cache/outputs+20251118_182042.npz --exp_vtip data/exp_data_ringA_Vtip+.csv --exp_htip data/exp_data_z_ringA_Vtip+.csv --cpd $(CPD) --fixed_vtip 4 --fixed_htip 8
dias1E11:
	python ring_radius_diamond_fitting.py --cache _ring_radius_cache/outputs_diamond+20251120_141925.npz --exp_vtip diamond_data_B888/exp_data_diamond_B888.csv --exp_htip diamond_data_B888/exp_data_z_diamond_B888.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 3
dias0: 
	python ring_radius_diamond_fitting.py --cache _ring_radius_cache/outputs_diamond+20251105_180001.npz --exp_vtip diamond_data_6B3645/exp_data_diamond_6B3645.csv --exp_htip diamond_data_6B3645/exp_data_z_diamond_6B3645.csv --cpd $(CPD) --fixed_vtip -2 --fixed_htip 3