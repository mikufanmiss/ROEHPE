fcst_steps=$((24*7))
observe_steps=$((2 * fcst_steps))
window_width=$((fcst_steps + observe_steps))

python example_benchmarks.py -d GEFCom --add_external_feature --observe_steps $observe_steps --zone_list 'CT' 'ME' 'NEMASSBOST' 'NH' 'RI' 'SEMASS' 'TOTAL' 'VT' 'WCMASS' --window_width $window_width --gpu 0 --avg_terms_list 1 --run_times 10 --persistence loop