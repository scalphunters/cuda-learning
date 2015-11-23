/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2004 FIMAT Group
 Copyright (C) 2007, 2009, 2010, 2011 StatPro Italia srl

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

#include <fincu/time/calendars/hongkong.hpp>
#include <fincu/errors.hpp>

namespace FinCu {

    HongKong::HongKong(Market m) {
        // all calendar instances share the same implementation instance
        static std::shared_ptr<Calendar::Impl> impl(new HongKong::HkexImpl);
        switch (m) {
          case HKEx:
            impl_ = impl;
            break;
          default:
            FC_FAIL("unknown market");
        }
    }

    bool HongKong::HkexImpl::isBusinessDay(const Date& date) const {
        Weekday w = date.weekday();
        Day d = date.dayOfMonth(), dd = date.dayOfYear();
        Month m = date.month();
        Year y = date.year();
        Day em = easterMonday(y);

        if (isWeekend(w)
            // New Year's Day
            || ((d == 1 || ((d == 2 || d == 3) && w == Monday))
                && m == January)
            // Good Friday
            || (dd == em-3)
            // Easter Monday
            || (dd == em)
            // Labor Day
            || ((d == 1 || ((d == 2 || d == 3) && w == Monday)) && m == May)
            // SAR Establishment Day
            || ((d == 1 || ((d == 2 || d == 3) && w == Monday)) && m == July)
            // National Day
            || ((d == 1 || ((d == 2 || d == 3) && w == Monday))
                && m == October)
            // Christmas Day
            || (d == 25 && m == December)
            // Boxing Day
            || (d == 26 && m == December)
            )
            return false;

        if (y == 2004) {
            if (// Lunar New Year
                ((d==22 || d==23 || d==24) && m == January)
                // Ching Ming Festival
                || (d == 5 && m == April)
                // Buddha's birthday
                || (d == 26 && m == May)
                // Tuen NG festival
                || (d == 22 && m == June)
                // Mid-autumn festival
                || (d == 29 && m == September)
                // Chung Yeung
                || (d == 29 && m == September))
                return false;
        }

        if (y == 2005) {
            if (// Lunar New Year
                ((d==9 || d==10 || d==11) && m == February)
                // Ching Ming Festival
                || (d == 5 && m == April)
                // Buddha's birthday
                || (d == 16 && m == May)
                // Tuen NG festival
                || (d == 11 && m == June)
                // Mid-autumn festival
                || (d == 19 && m == September)
                // Chung Yeung festival
                || (d == 11 && m == October))
            return false;
        }

        if (y == 2006) {
            if (// Lunar New Year
                ((d >= 28 && d <= 31) && m == January)
                // Ching Ming Festival
                || (d == 5 && m == April)
                // Buddha's birthday
                || (d == 5 && m == May)
                // Tuen NG festival
                || (d == 31 && m == May)
                // Mid-autumn festival
                || (d == 7 && m == October)
                // Chung Yeung festival
                || (d == 30 && m == October))
            return false;
        }

        if (y == 2007) {
            if (// Lunar New Year
                ((d >= 17 && d <= 20) && m == February)
                // Ching Ming Festival
                || (d == 5 && m == April)
                // Buddha's birthday
                || (d == 24 && m == May)
                // Tuen NG festival
                || (d == 19 && m == June)
                // Mid-autumn festival
                || (d == 26 && m == September)
                // Chung Yeung festival
                || (d == 19 && m == October))
            return false;
        }

        if (y == 2008) {
            if (// Lunar New Year
                ((d >= 7 && d <= 9) && m == February)
                // Ching Ming Festival
                || (d == 4 && m == April)
                // Buddha's birthday
                || (d == 12 && m == May)
                // Tuen NG festival
                || (d == 9 && m == June)
                // Mid-autumn festival
                || (d == 15 && m == September)
                // Chung Yeung festival
                || (d == 7 && m == October))
            return false;
        }

        if (y == 2009) {
            if (// Lunar New Year
                ((d >= 26 && d <= 28) && m == January)
                // Ching Ming Festival
                || (d == 4 && m == April)
                // Buddha's birthday
                || (d == 2 && m == May)
                // Tuen NG festival
                || (d == 28 && m == May)
                // Mid-autumn festival
                || (d == 3 && m == October)
                // Chung Yeung festival
                || (d == 26 && m == October))
            return false;
        }

        if (y == 2010) {
            if (// Lunar New Year
                ((d == 15 || d == 16) && m == February)
                // Ching Ming Festival
                || (d == 6 && m == April)
                // Buddha's birthday
                || (d == 21 && m == May)
                // Tuen NG festival
                || (d == 16 && m == June)
                // Mid-autumn festival
                || (d == 23 && m == September))
            return false;
        }

        if (y == 2011) {
            if (// Lunar New Year
                ((d == 3 || d == 4) && m == February)
                // Ching Ming Festival
                || (d == 5 && m == April)
                // Buddha's birthday
                || (d == 10 && m == May)
                // Tuen NG festival
                || (d == 6 && m == June)
                // Mid-autumn festival
                || (d == 13 && m == September)
                // Chung Yeung festival
                || (d == 5 && m == October)
                // Second day after Christmas
                || (d == 27 && m == December))
            return false;
        }

        if (y == 2012) {
            if (// Lunar New Year
                (d >= 23 && d <= 25 && m == January)
                // Ching Ming Festival
                || (d == 4 && m == April)
                // Buddha's birthday
                || (d == 10 && m == May)
                // Mid-autumn festival
                || (d == 1 && m == October)
                // Chung Yeung festival
                || (d == 23 && m == October))
            return false;
        }

        if (y == 2013) {
            if (// Lunar New Year
                (d >= 11 && d <= 13 && m == February)
                // Ching Ming Festival
                || (d == 4 && m == April)
                // Buddha's birthday
                || (d == 17 && m == May)
                // Tuen Ng festival
                || (d == 12 && m == June)
                // Mid-autumn festival
                || (d == 20 && m == September)
                // Chung Yeung festival
                || (d == 14 && m == October))
            return false;
        }

        if (y == 2014) {
            if (// Lunar New Year
                ((d == 31 && m == January) || (d <= 3 && m == February))
                // Buddha's birthday
                || (d == 6 && m == May)
                // Tuen Ng festival
                || (d == 2 && m == June)
                // Mid-autumn festival
                || (d == 9 && m == September)
                // Chung Yeung festival
                || (d == 2 && m == October))
            return false;
        }

        if (y == 2015) {
            if (// Lunar New Year
                ((d == 19 && m == February) || (d == 20 && m == February))
                // The day following Easter Monday
                || (d == 7 && m == April)
                // Buddha's birthday
                || (d == 25 && m == May)
                // Tuen Ng festival
                || (d == 20 && m == June)
                // Mid-autumn festival
                || (d == 28 && m == September)
                // Chung Yeung festival
                || (d == 21 && m == October))
                return false;
        }

        return true;
    }

}
