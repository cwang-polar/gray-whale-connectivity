"""Scientific boundary checks for the isolated effort-rate diagnostic."""

import unittest
import numpy as np
import pandas as pd
from shapely.geometry import box
from gray_whale_corridor.metrics.survey_effort import (
    aggregate_flights, clipped_fractions, flight_bootstrap, reconstruct_segments, resolve_modes, summarize_rates,
)


def events():
    return pd.DataFrame({
        'flight_key':['a']*4+['b']*2,
        'datetime':pd.to_datetime(['2011-07-01 10:00','2011-07-01 10:01','2011-07-01 10:02',
                                   '2011-07-01 10:03','2011-07-01 11:00','2011-07-01 11:01']),
        'event_number':[1,2,3,4,1,2], 'source_id':['a1','a2','a3','a4','b1','b2'],
        'FltType':[2,-1,4,-1,-1,2], 'entry_clean':['start transect','.','end transect','.','.','start transect'],
        'lat':[70]*6,'lon':[-160,-159.99,-159.98,-159.97,-160,-159.99],
        'year':[2011]*6,'month':[7]*6,
    })


class EffortTests(unittest.TestCase):
    def test_automatic_modes_do_not_cross_flights(self):
        out=resolve_modes(events())
        self.assertEqual(out.loc[:3,'mode'].tolist(),[2,2,4,4])
        self.assertTrue(np.isnan(out.loc[4,'mode']))

    def test_boundary_uses_mode_before_transition(self):
        out=reconstruct_segments(events())
        self.assertEqual(len(out),4)
        self.assertEqual(out.loc[:2,'start_mode'].tolist(),[2,2,4])
        self.assertEqual(out.accepted.tolist(),[True,True,True,False])

    def test_long_gap_excluded(self):
        data=events().iloc[:2].copy()
        data.loc[1,'datetime']=data.loc[0,'datetime']+pd.Timedelta(minutes=20)
        self.assertFalse(reconstruct_segments(data).accepted.iloc[0])

    def test_polygon_clips_distance_fraction(self):
        seg=pd.DataFrame({'start_x':[0.], 'start_y':[0.], 'x':[10.], 'y':[0.],
                          'valid_location':[True], 'valid_time':[True]})
        self.assertAlmostEqual(clipped_fractions(seg,box(0,-1,5,1))[0],.5)

    def test_zero_sighting_flights_retained(self):
        seg=reconstruct_segments(events().iloc[:4])
        seg['fraction']=1.
        obs=resolve_modes(events().iloc[:0]).assign(member=pd.Series(dtype=bool),
            repeat_clean=pd.Series(dtype=str),whale_count=pd.Series(dtype=float))
        flights=aggregate_flights(seg,obs,fraction_column='fraction',member_column='member',modes=[2])
        self.assertEqual(len(flights),1)
        self.assertEqual(flights.sighting_count.iloc[0],0)
        self.assertGreater(flights.survey_km.iloc[0],0)
        self.assertEqual(summarize_rates(flights,[2011]).whales_per_100km.iloc[0],0)

    def test_missing_effort_is_not_zero_whale_use(self):
        flights=pd.DataFrame({'flight_key':['a'],'year':[2011],'survey_km':[0.],
          'survey_hours':[0.],'sighting_count':[0.],'reported_whales':[0.],
          'missing_group_counts':[0],'unmatched_numerator':[False]})
        self.assertTrue(np.isnan(summarize_rates(flights,[2011]).whales_per_100km.iloc[0]))

    def test_numerator_uses_same_mode_and_excludes_confirmed_repeats(self):
        seg=reconstruct_segments(events().iloc[:4])
        seg['fraction']=1.
        obs=resolve_modes(events().iloc[:3]).assign(member=True,repeat_clean=['yes','no','no'],whale_count=[4.,2.,3.])
        flights=aggregate_flights(seg,obs,fraction_column='fraction',member_column='member',modes=[2])
        self.assertEqual(flights.reported_whales.iloc[0],2.)
        self.assertEqual(flights.sighting_count.iloc[0],1.)

    def test_clipped_time_and_distance_share_spatial_fraction(self):
        seg=reconstruct_segments(events().iloc[:2])
        seg['fraction']=.5
        obs=resolve_modes(events().iloc[:0]).assign(member=pd.Series(dtype=bool),repeat_clean=pd.Series(dtype=str),whale_count=pd.Series(dtype=float))
        flights=aggregate_flights(seg,obs,fraction_column='fraction',member_column='member',modes=[2])
        self.assertAlmostEqual(flights.survey_km.iloc[0],seg.raw_km.iloc[0]*.5)
        self.assertAlmostEqual(flights.survey_hours.iloc[0],.5/60)

    def test_missing_group_counts_do_not_condition_bootstrap_on_complete_flights(self):
        flights=pd.DataFrame({'year':[2011,2011], 'month':[7,7],
            'reported_whales':[np.nan,2.], 'sighting_count':[1.,1.],
            'survey_km':[10.,20.], 'survey_hours':[.1,.2]})
        ci=flight_bootstrap(flights,replicates=40,seed=1).set_index('metric')
        self.assertEqual(ci.loc['whales_per_100km','valid_replicates'],0)
        self.assertTrue(np.isnan(ci.loc['whales_per_100km','lower']))
        self.assertEqual(ci.loc['sightings_per_100km','valid_replicates'],40)


if __name__=='__main__':
    unittest.main()
