/** \file "SolarPosition.cpp" Primary source file for solar position model plug-in.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SolarPosition.h"

using namespace std;
using namespace helios;

SolarPosition::SolarPosition( helios::Context* __context ){
  context = __context;
  UTC = 8;
  latitude = 38.55;
  longitude = 121.76;
}


SolarPosition::SolarPosition( int UTC_hrs, float latitude_deg, float longitude_deg, helios::Context* context_ptr ){
  context = context_ptr;
  UTC = UTC_hrs;
  latitude = latitude_deg;
  longitude = longitude_deg;

  if( latitude_deg<-90 || latitude_deg>90 ){
    std::cout << "WARNING (SolarPosition): Latitude must be between -90 and +90 deg (a latitude of " << latitude << " was given). Default latitude is being used." << std::endl;
    latitude = 38.55;
  }else{
    latitude = latitude_deg;
  }
  
  if( longitude<-180 || longitude>180 ){
    std::cout << "WARNING (SolarPosition): Longitude must be between -180 and +180 deg (a longitude of " << longitude << " was given). Default longitude is being used." << std::endl;
    longitude = 121.76;
  }else{
    longitude = longitude_deg;
  }
    
}

int SolarPosition::selfTest() const{

  std::cout << "Running solar position model self-test..." << std::flush;
  int error_count = 0;

  float latitude;
  float longitude;
  Date date;
  Time time;
  int UTC;

  std::string answer;

  float errtol = 1e-6;

  Context context_s;

  //---- Sun Zenith/Azimuth Test for Boulder, CO ---- //

  latitude = 40.1250;
  longitude = 105.2369;
  date = make_Date(1,1,2000);
  time = make_Time(10,30,0) ;
  UTC = 7;

  float theta_actual = 29.49;
  float phi_actual = 154.18;  

  context_s.setDate( date );
  context_s.setTime( time );

  SolarPosition solarposition_1( UTC, latitude, longitude, &context_s );

  float theta_s = solarposition_1.getSunElevation()*180.f/M_PI;
  float phi_s = solarposition_1.getSunAzimuth()*180.f/M_PI;

  if( fabs(theta_s-theta_actual)>10 || fabs(phi_s-phi_actual)>5 ){
    error_count++;
    std::cout << "failed: verification test for known solar position does not agree with calculated result." << std::endl;
  } 

  //---- Test of Gueymard Solar Flux Model ---- //

  latitude = 36.5289; //Billings, OK
  longitude = 97.4439;
  date = make_Date(5,5,2003);
  time = make_Time(9,10,0) ;
  UTC = 6;

  context_s.setDate( date );
  context_s.setTime( time );

  SolarPosition solarposition_2( UTC, latitude, longitude, &context_s );

  float pressure = 96660;
  float temperature = 290;
  float humidity = 0.5;
  float turbidity = 0.025;

  float Eb, Eb_PAR, Eb_NIR, fdiff;

  solarposition_2.GueymardSolarModel( pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff );
  Eb=Eb_PAR+Eb_NIR;

  //----- Test of Ambient Longwave Model ------ //

  SolarPosition solarposition_3( UTC, latitude, longitude, &context_s );

  temperature = 290;
  humidity = 0.5;

  float LW = solarposition_3.getAmbientLongwaveFlux( temperature, humidity );

  if( fabs(LW-310.03192f)>errtol ){
    error_count++;
    std::cout << "failed: verification test for ambient longwave model does not agree with known result." << std::endl;
  }

  if( error_count==0 ){
    std::cout << "passed." << std::endl;
    return 0;
  }else{
    std::cout << "Failed Context self-test with " << error_count << " errors." << std::endl;
    return 1;
  }



}

SphericalCoord SolarPosition::calculateSunDirection( const helios::Time &time, const helios::Date &date ) const{
  
  int solstice_day, LSTM;
  float Gamma, delta, time_dec, B, EoT, TC, LST, h, theta, phi, rad;

  rad=M_PI/180.f;

  solstice_day=81;

  //day angle (Iqbal Eq. 1.1.2)
  Gamma = 2.f*M_PI*(float(date.JulianDay()-1))/365.f;  

  //solar declination angle (Iqbal Eq. 1.3.1 after Spencer)
  delta = 0.006918f - 0.399912f*cos(Gamma) + 0.070257f*sin(Gamma) 
    - 0.006758f*cos(2.f*Gamma) + 0.000907f*sin(2.f*Gamma) 
    - 0.002697f*cos(3.f*Gamma) + 0.00148f*sin(3.f*Gamma);

  //equation of time (Iqbal Eq. 1.4.1 after Spencer)
  EoT = 229.18f*(0.000075f + 0.001868f*cos(Gamma) - 0.032077f*sin(Gamma)
		 - 0.014615f*cos(2.f*Gamma) - 0.04089f*sin(2.f*Gamma));

  time_dec=time.hour+time.minute/60.f;  //(hours) 

  LSTM=15.f*float(UTC); //degrees

  TC=4.f*(LSTM-longitude)+EoT; //minutes
  LST=time_dec+TC/60.f; //hours

  h=(LST-12.f)*15.f*rad; //hour angle (rad)
    
  //solar zentih angle
  theta = asin_safe( sin(latitude*rad)*sin(delta) + cos(latitude*rad)*cos(delta)*cos(h) ); //(rad)

  assert( theta>-0.5f*M_PI && theta<0.5f*M_PI );

  //solar elevation angle
  phi = acos_safe( (sin(delta) - sin(theta)*sin(latitude*rad))/(cos(theta)*cos(latitude*rad)));

  if( LST>12.f ){
    phi=2.f*M_PI-phi;
  }

  assert( phi>0 && phi<2.f*M_PI );

  return make_SphericalCoord(theta,phi);

}

Time SolarPosition::getSunriseTime() const{

  //This is a lazy way to find the sunrise/sunset time.  If anyone wants to do the math and solve for it algebraically, go for it.
  SphericalCoord sun_dir;
  for( uint h=1; h<=23; h++ ){
    for( uint m=1; m<=59; m++ ){
      SphericalCoord sun_dir = calculateSunDirection(make_Time(h,m,0),context->getDate());
      if( sun_dir.elevation>0 ){
  	return make_Time(h,m);
      }
    }
  }

  return make_Time(0,0); //should never get here

}

Time SolarPosition::getSunsetTime() const{

  // SphericalCoord sun_dir;
  for( uint h=23; h>=1; h-- ){
    for( uint m=59; m>=1; m-- ){
      SphericalCoord sun_dir = calculateSunDirection(make_Time(h,m,0),context->getDate());
      if( sun_dir.elevation>0 ){
  	return make_Time(h,m);
      }
    }
  }

  return make_Time(0,0); //should never get here

}

float SolarPosition::getSunElevation() const{
    float elevation;
    if( issolarpositionoverridden ){
        elevation = sun_direction.elevation;
    }else{
        elevation = calculateSunDirection(context->getTime(),context->getDate()).elevation;
    }
    return elevation;
}

float SolarPosition::getSunZenith() const{
    float zenith;
    if( issolarpositionoverridden ){
        zenith = sun_direction.zenith;
    }else{
        zenith = calculateSunDirection(context->getTime(),context->getDate()).zenith;
    }
    return zenith;
}

float SolarPosition::getSunAzimuth() const{
    float azimuth;
    if( issolarpositionoverridden ){
        azimuth = sun_direction.azimuth;
    }else{
        azimuth = calculateSunDirection(context->getTime(),context->getDate()).azimuth;
    }
    return azimuth;
}

vec3 SolarPosition::getSunDirectionVector() const{
    SphericalCoord sundirection;
    if( issolarpositionoverridden ) {
        sundirection = sun_direction;
    }else{
        sundirection = calculateSunDirection(context->getTime(),context->getDate());
    }
    return sphere2cart(sundirection);
}

SphericalCoord SolarPosition::getSunDirectionSpherical() const{
    SphericalCoord sundirection;
    if( issolarpositionoverridden ) {
        sundirection = sun_direction;
    }else{
        sundirection = calculateSunDirection(context->getTime(),context->getDate());
    }
    return sundirection;
}

void SolarPosition::setSunDirection( const helios::SphericalCoord &sundirection ){
    issolarpositionoverridden = true;
    sun_direction = sundirection;
}

float SolarPosition::getSolarFlux(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity ) const{
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    float Eb = Eb_PAR+Eb_NIR;
    if( !cloudcalibrationlabel.empty() ){
        applyCloudCalibration( Eb, fdiff );
    }
    return Eb;
}

float SolarPosition::getSolarFluxPAR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity ) const{
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if( !cloudcalibrationlabel.empty() ){
        applyCloudCalibration( Eb_PAR, fdiff );
    }
    return Eb_PAR;
}

float SolarPosition::getSolarFluxNIR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity ) const{
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if( !cloudcalibrationlabel.empty() ){
        applyCloudCalibration( Eb_NIR, fdiff );
    }
    return Eb_NIR;
}

float SolarPosition::getDiffuseFraction(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity ) const{
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if( !cloudcalibrationlabel.empty() ){
        applyCloudCalibration( Eb_PAR, fdiff );
    }
    return fdiff;
}

void SolarPosition::GueymardSolarModel( float pressure, float temperature, float humidity, float turbidity, float& Eb_PAR, float& Eb_NIR, float &fdiff ) const{

  float beta = turbidity;

  uint DOY = context->getJulianDate();

  float theta = getSunZenith();

  if( theta>0.5*M_PI ){
    Eb_PAR = 0.f;
    Eb_NIR = 0.f;
    fdiff = 1.f;
    return;
  }
  
  float m=pow(cos(theta)+0.15*pow(93.885-theta*180/M_PI,-1.25),-1);

  float E0_PAR=635.4;
  float E0_NIR=709.7;

  vec2 alpha(1.3,1.3);

  //---- Rayleigh ----//
  //NOTE: Rayleigh scattering dominates the atmospheric attenuation, and thus variations in the model predictions are almost entirely due to pressure (and theta)
  float mR=1.f/(cos(theta)+0.48353*pow(theta*180/M_PI,0.095846)/pow(96.741-theta*180/M_PI,1.754));

  float mR_p=mR*pressure/101325;

  float TR_PAR=(1.f+1.8169*mR_p-0.033454*mR_p*mR_p)/(1.f+2.063*mR_p+0.31978*mR_p*mR_p);

  float TR_NIR=(1.f-0.010394*mR_p)/(1.f-0.00011042*mR_p*mR_p);

  //---- Uniform gasses ----//
  mR=1.f/(cos(theta)+0.48353*pow(theta*180/M_PI,0.095846)/pow(96.741-theta*180/M_PI,1.754));

  mR_p=mR*pressure/101325;

  float Tg_PAR=(1.f+0.95885*mR_p+0.012871*mR_p*mR_p)/(1.f+0.96321*mR_p+0.015455*mR_p*mR_p);

  float Tg_NIR=(1.f+0.27284*mR_p-0.00063699*mR_p*mR_p)/(1.f+0.30306*mR_p);

  float BR_PAR=0.5f*(0.89013-0.0049558*mR+0.000045721*mR*mR);

  float BR_NIR=0.5;

  float Ba=1.f-exp(-0.6931-1.8326*cos(theta));

  //---- Ozone -----//
  float uo=(235+(150+40*sin(0.9856*(DOY-30)*M_PI/180.f)+20*sin(3*(longitude*M_PI/180.f+20)))*pow(sin(1.28*latitude*M_PI/180.f),2))*0.001f; //O3 atm-cm
  //NOTE: uo model from van Heuklon (1979)
  float mo=m;

  float f1=uo*(10.979-8.5421*uo)/(1.f+2.0115*uo+40.189*uo*uo);
  float f2=uo*(-0.027589-0.005138*uo)/(1.f-2.4857*uo+13.942*uo*uo);
  float f3=uo*(10.995-5.5001*uo)/(1.f+1.678*uo+42.406*uo*uo);
  float To_PAR=(1.f+f1*mo+f2*mo*mo)/(1.f+f3*mo);

  float To_NIR=1.f;

  //---- Nitrogen ---- //
  float un=0.0002; //N atm-cm
  float mw=1.f/(cos(theta)+1.065*pow(theta*180/M_PI,1.6132)/pow(111.55-theta*180/M_PI,3.2629));

  float g1=(0.17499+41.654*un-2146.4*un*un)/(1+22295*un*un);
  float g2=un*(-1.2134+59.324*un)/(1.f+8847.8*un*un);
  float g3=(0.17499+61.658*un+9196.4*un*un)/(1.f+74109*un*un);
  float Tn_PAR=fmin(1.f,(1.f+g1*mw+g2*mw*mw)/(1.f+g3*mw));
  float Tn_PAR_p=fmin(1.f,(1.f+g1*1.66+g2*1.66*1.66)/(1.f+g3*1.66));

  float Tn_NIR=1.f;
  float Tn_NIR_p=1.f;

  //---- Water -----//
  float gamma=log(humidity)+17.67*(temperature-273)/(243+25);
  float tdp=243*gamma/(17.67-gamma)*9/5+32;//dewpoint temperature in Fahrenheit
  float w=exp((0.1133-log(4.0+1))+0.0393*tdp); //cm of precipitable water
  //NOTE: precipitable water model from Viswanadham (1981), Eq. 5
  mw=1.f/(cos(theta)+1.1212*pow(theta*180/M_PI,0.6379)/pow(93.781-theta*180/M_PI,1.9203));

  float h1=w*(0.065445+0.00029901*w)/(1.f+1.2728*w);
  float h2=w*(0.065687+0.0013218*w)/(1.f+1.2008*w);
  float Tw_PAR=(1.f+h1*mw)/(1.f+h2*mw);
  float Tw_PAR_p=(1.f+h1*1.66)/(1.f+h2*1.66);

  float c1=w*(19.566-1.6506*w+1.0672*w*w)/(1.f+5.4248*w+1.6005*w*w);
  float c2=w*(0.50158-0.14732*w+0.047584*w*w)/(1.f+1.1811*w+1.0699*w*w);
  float c3=w*(21.286-0.39232*w+1.2692*w*w)/(1.f+4.8318*w+1.412*w*w);
  float c4=w*(0.70992-0.23155*w+0.096541*w*w)/(1.f+0.44907*w+0.75425*w*w);
  float Tw_NIR=(1.f+c1*mw+c2*mw*mw)/(1.f+c3*mw+c4*mw*mw);
  float Tw_NIR_p=(1.f+c1*1.66+c2*1.66*1.66)/(1.f+c3*1.66+c4*1.66*1.66);

  //---- Aerosol ----//
  float ma=1.f/(cos(theta)+0.16851*pow(theta*180/M_PI,0.18198)/pow(95.318-theta*180/M_PI,1.9542));
  float ua=log(1.f+ma*beta);

  float d0=0.57664-0.024743*alpha.x;
  float d1=(0.093942-0.2269*alpha.x+0.12848*alpha.x*alpha.x)/(1.f+0.6418*alpha.x);
  float d2=(-0.093819+0.36668*alpha.x-0.12775*alpha.x*alpha.x)/(1.f-0.11651*alpha.x);
  float d3=alpha.x*(0.15232-0.08721*alpha.x+0.012664*alpha.x*alpha.x)/(1.f-0.90454*alpha.x+0.26167*alpha.x*alpha.x);
  float lambdae_PAR=(d0+d1*ua+d2*ua*ua)/(1.f+d3*ua*ua);
  float Ta_PAR=exp(-ma*beta*pow(lambdae_PAR,-alpha.x));

  float e0 = (1.183-0.022989*alpha.y+0.020829*alpha.y*alpha.y)/(1.f+0.11133*alpha.y);
  float e1 = (-0.50003-0.18329*alpha.y+0.23835*alpha.y*alpha.y)/(1.f+1.6756*alpha.y);
  float e2 = (-0.50001+1.1414*alpha.y+0.0083589*alpha.y*alpha.y)/(1.f+11.168*alpha.y);
  float e3 = (-0.70003-0.73587*alpha.y+0.51509*alpha.y*alpha.y)/(1.f+4.7665*alpha.y);
  float lambdae_NIR = (e0+e1*ua+e2*ua*ua)/(1.f+e3*ua);
  float Ta_NIR=exp(-ma*beta*pow(lambdae_NIR,-alpha.y));

  float omega_PAR=1.0;
  float omega_NIR=1.0;

  float Tas_PAR=exp(-ma*omega_PAR*beta*pow(lambdae_PAR,-alpha.x));

  float Tas_NIR=exp(-ma*omega_NIR*beta*pow(lambdae_NIR,-alpha.y));

  //direct irradiation
  Eb_PAR=TR_PAR*Tg_PAR*To_PAR*Tn_PAR*Tw_PAR*Ta_PAR*E0_PAR;
  Eb_NIR=TR_NIR*Tg_NIR*To_NIR*Tn_NIR*Tw_NIR*Ta_NIR*E0_NIR;
  float Eb=Eb_PAR+Eb_NIR;

  //diffuse irradiation
  float Edp_PAR=To_PAR*Tg_PAR*Tn_PAR_p*Tw_PAR_p*(BR_PAR*(1.f-TR_PAR)*pow(Ta_PAR,0.25)+Ba*TR_PAR*(1.f-pow(Tas_PAR,0.25)))*E0_PAR;
  float Edp_NIR=To_NIR*Tg_NIR*Tn_NIR_p*Tw_NIR_p*(BR_NIR*(1.f-TR_NIR)*pow(Ta_NIR,0.25)+Ba*TR_NIR*(1.f-pow(Tas_NIR,0.25)))*E0_NIR;
  float Edp=Edp_PAR+Edp_NIR;

  //diffuse fraction
  fdiff = Edp/(Eb+Edp);

  //fraction can't be greater than 1.0
  if( fdiff>1.0 ){
    fdiff=1;
  }

  //fraction can't be less than 0
  if( fdiff<0 ){
    fdiff=0;
  }

    
}

float SolarPosition::getAmbientLongwaveFlux(float temperature_K, float humidity_rel ) const{

  //Model from Prata (1996) Q. J. R. Meteorol. Soc.

  float e0 = 611.f * exp(17.502f * (temperature_K - 273.f) / ((temperature_K - 273.f) + 240.9f)) * humidity_rel; //Pascals

  float K = 0.465f; //cm-K/Pa

  float xi = e0 / temperature_K * K;
  float eps = 1.f-(1.f+xi)*exp(-sqrt(1.2f+3.f*xi));

  return eps*5.67e-8*pow(temperature_K, 4);

}

float turbidityResidualFunction(float turbidity, std::vector<float> &parameters, const void * a_solarpositionmodel) {

    auto* solarpositionmodel = reinterpret_cast<const SolarPosition*>(a_solarpositionmodel);

    float pressure = parameters.at(0);
    float temperature = parameters.at(1);
    float humidity = parameters.at(2);
    float flux_target = parameters.at(3);

    float flux_model = solarpositionmodel->getSolarFlux( pressure, temperature, humidity, turbidity )*cosf(solarpositionmodel->getSunZenith());
    return flux_model-flux_target;
}

float SolarPosition::calibrateTurbidityFromTimeseries( const std::string &timeseries_shortwave_flux_label_Wm2 ) const{

    if( !context->doesTimeseriesVariableExist(timeseries_shortwave_flux_label_Wm2.c_str()) ){
        helios_runtime_error("ERROR (SolarPosition::calibrateTurbidityFromTimeseries): Timeseries variable " + timeseries_shortwave_flux_label_Wm2 + " does not exist.");
    }

    uint length = context->getTimeseriesLength(timeseries_shortwave_flux_label_Wm2.c_str() );

    float min_flux = 1e6;
    float max_flux = 0;
    int max_flux_index = 0;
    for( int t=0; t<length; t++ ){
        float flux = context->queryTimeseriesData(timeseries_shortwave_flux_label_Wm2.c_str(), t );
        if( flux<min_flux ){
            min_flux = flux;
        }
        if( flux>max_flux ){
            max_flux = flux;
            max_flux_index = t;
        }
    }

    if( max_flux<750 || max_flux>1200 ){
        helios_runtime_error("ERROR (SolarPosition::calibrateTurbidityFromTimeseries): The maximum flux for the timeseries data is not within the expected range. Either it is not solar flux data, or there are no clear sky days in the dataset");
    }else if( min_flux<0 ){
        helios_runtime_error("ERROR (SolarPosition::calibrateTurbidityFromTimeseries): The minimum flux for the timeseries data is negative. Solar fluxes cannot be negative.");
    }

    std::vector<float> parameters{ 101325, 300, 0.5, max_flux };

    SolarPosition solarposition_copy( UTC, latitude, longitude, context );
    Date date_max = context->queryTimeseriesDate(timeseries_shortwave_flux_label_Wm2.c_str(), max_flux_index );
    Time time_max = context->queryTimeseriesTime(timeseries_shortwave_flux_label_Wm2.c_str(), max_flux_index );

    solarposition_copy.setSunDirection( solarposition_copy.calculateSunDirection( time_max, date_max) );

    float turbidity = fzero( turbidityResidualFunction, parameters, &solarposition_copy, 0.01 );

    return turbidity;

}

void SolarPosition::enableCloudCalibration( const std::string &timeseries_shortwave_flux_label_Wm2 ){

    if( !context->doesTimeseriesVariableExist( timeseries_shortwave_flux_label_Wm2.c_str()) ){
        helios_runtime_error("ERROR (SolarPosition::enableCloudCalibration): Timeseries variable " + timeseries_shortwave_flux_label_Wm2 + " does not exist.");
    }
    
    cloudcalibrationlabel = timeseries_shortwave_flux_label_Wm2;
    
}

void SolarPosition::disableCloudCalibration(){
    cloudcalibrationlabel = "";
}

void SolarPosition::applyCloudCalibration(float &R_calc_Wm2, float &fdiff_calc) const{

    assert( context->doesTimeseriesVariableExist( cloudcalibrationlabel.c_str()) );

    float R_meas = context->queryTimeseriesData( cloudcalibrationlabel.c_str() );
    float R_calc_horiz = R_calc_Wm2*cosf(getSunZenith());

    float fdiff = fmin(fmax(0, 1.f-(R_meas - R_calc_horiz) / (R_calc_horiz)), 1);
    float R = R_calc_Wm2 * R_meas / R_calc_horiz;

    if( fdiff>0.001 && R_calc_horiz>1.f ){
        R_calc_Wm2 = R;
        fdiff_calc = fdiff;
    }

}
  
